# This file is modified from splaTAM,
# licensed under the BSD 3-Clause License.

import numpy as np
import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from slam.common.common import setup_camera
from slam.model_components.slam_external_splatam import (
    accumulate_mean2d_gradient, build_rotation)
from slam.model_components.slam_helpers_splatam import (
    transform_to_frame, transformed_params2depthplussilhouette,
    transformed_params2rendervar)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianCloud(nn.Module):
    def __init__(self, init_rgb, init_depth, w2c, camera, prune_dict,
                 densify_dict):
        super(GaussianCloud, self).__init__()
        self.camera = camera
        self.gaussian_cam = setup_camera(self.camera,
                                         w2c.detach().cpu().numpy())
        self.first_frame_w2c = w2c.detach()
        self.prune_dict = prune_dict
        self.densify_dict = densify_dict
        self.params, self.variables = self.initialize_gaussians(
            init_rgb, init_depth, w2c)
        self.variables['scene_radius'] = init_depth.max() / 3.0

    def initialize_gaussians(self, init_rgb, init_depth, w2c):
        mask = (init_depth > 0)  # Mask out invalid depth values
        mask = mask.reshape(-1)
        init_pt_cld, mean3_sq_dist = self.get_pointcloud(
            color=init_rgb,
            depth=init_depth,
            w2c=w2c,
            mask=mask,
            compute_mean_sq_dist=True)
        params, variables = self.initialize_params(init_pt_cld, mean3_sq_dist)
        return params, variables

    def render(self,
               w2c,
               gaussians_grad=False,
               camera_grad=False,
               retain_grad=True):
        transformed_pts = transform_to_frame(self.params['means3D'],
                                             w2c,
                                             gaussians_grad=gaussians_grad,
                                             camera_grad=camera_grad)
        # Initialize Render Variables
        rendervar = transformed_params2rendervar(self.params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            self.params, self.first_frame_w2c, transformed_pts)
        # RGB Rendering
        if retain_grad:
            rendervar['means2D'].retain_grad()
        im, radius, depth, = Renderer(
            raster_settings=self.gaussian_cam)(**rendervar)  # [C, H, W]
        if retain_grad:
            self.variables['means2D'] = rendervar['means2D']
        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=self.gaussian_cam)(
            **depth_sil_rendervar)
        # get outputs
        outputs = {'rgb': im, 'depth_sil': depth_sil, 'depth': depth}

        seen = radius > 0
        self.variables['max_2D_radius'][seen] = torch.max(
            radius[seen], self.variables['max_2D_radius'][seen])
        self.variables['seen'] = seen

        return outputs

    def remove_points(self, to_remove, optimizer):
        to_keep = ~to_remove
        for param_group, cur_optimizer in optimizer.items():
            if 'pose' not in param_group:
                group = cur_optimizer.param_groups[0]
                params_to_update = group['params'][0]
                # Update state for exp_avg and exp_avg_sq
                stored_state = cur_optimizer.state.get(params_to_update, None)
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][to_keep]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][
                        to_keep]
                    del cur_optimizer.state[params_to_update]
                    cur_optimizer.state[group['params'][0]] = stored_state
                # Update parameters
                group['params'][0] = torch.nn.Parameter(
                    (params_to_update[to_keep].requires_grad_(True)))

                self.params[param_group] = group['params'][0]

        self.variables['means2D_gradient_accum'] = self.variables[
            'means2D_gradient_accum'][to_keep]
        self.variables['denom'] = self.variables['denom'][to_keep]
        self.variables['max_2D_radius'] = self.variables['max_2D_radius'][
            to_keep]
        if 'timestep' in self.variables.keys():
            self.variables['timestep'] = self.variables['timestep'][to_keep]

    def update_params_and_optimizer(self, new_params, optimizer):
        for k, v in new_params.items():
            cur_optimizer = optimizer[k]
            group = cur_optimizer.param_groups[0]
            stored_state = cur_optimizer.state.get(group['params'][0], None)
            stored_state['exp_avg'] = torch.zeros_like(v)
            stored_state['exp_avg_sq'] = torch.zeros_like(v)
            del optimizer.state[group['params'][0]]
            group['params'][0] = torch.nn.Parameter(v.requires_grad_(True))
            cur_optimizer.state[group['params'][0]] = stored_state
            self.params[k] = group['params'][0]

    def prune_gaussians(self, iter, optimizer):
        if iter <= self.prune_dict['stop_after']:
            if (iter >= self.prune_dict['start_after']) and (
                    iter % self.prune_dict['prune_every'] == 0):
                if iter == self.prune_dict['stop_after']:
                    remove_threshold = self.prune_dict[
                        'final_removal_opacity_threshold']
                else:
                    remove_threshold = self.prune_dict[
                        'removal_opacity_threshold']
                # Remove Gaussians with low opacity
                to_remove = (torch.sigmoid(self.params['logit_opacities']) <
                             remove_threshold).squeeze()
                # Remove Gaussians that are too big
                if iter >= self.prune_dict['remove_big_after']:
                    big_points_ws = torch.exp(self.params['log_scales']).max(
                        dim=1).values > 0.1 * self.variables['scene_radius']
                    to_remove = torch.logical_or(to_remove, big_points_ws)
                self.remove_points(to_remove, optimizer)
                torch.cuda.empty_cache()

            # Reset Opacities for all Gaussians
            if iter > 0 and iter % self.prune_dict[
                    'reset_opacities_every'] == 0 and self.prune_dict[
                        'reset_opacities']:
                new_params = {
                    'logit_opacities':
                    inverse_sigmoid(
                        torch.ones_like(self.params['logit_opacities']) * 0.01)
                }
                self.update_params_and_optimizer(new_params, optimizer)

    def cat_params_to_optimizer(self, new_params, optimizer):
        for k, v in new_params.items():
            cur_optimizer = optimizer[k]
            group = cur_optimizer.param_groups[0]
            stored_state = cur_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat(
                    (stored_state['exp_avg'], torch.zeros_like(v)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat(
                    (stored_state['exp_avg_sq'], torch.zeros_like(v)), dim=0)
                del cur_optimizer.state[group['params'][0]]
                group['params'][0] = torch.nn.Parameter(
                    torch.cat((group['params'][0], v),
                              dim=0).requires_grad_(True))
                cur_optimizer.state[group['params'][0]] = stored_state
                self.params[k] = group['params'][0]
            else:
                group['params'][0] = torch.nn.Parameter(
                    torch.cat((group['params'][0], v),
                              dim=0).requires_grad_(True))
                self.params[k] = group['params'][0]

    def densify(self, iter, optimizer):
        if iter <= self.densify_dict['stop_after']:
            self.variables = accumulate_mean2d_gradient(self.variables)
            grad_thresh = self.densify_dict['grad_thresh']
            if (iter >= self.densify_dict['start_after']) and (
                    iter % self.densify_dict['densify_every'] == 0):
                grads = self.variables[
                    'means2D_gradient_accum'] / self.variables['denom']
                grads[grads.isnan()] = 0.0
                to_clone = torch.logical_and(grads >= grad_thresh, (torch.max(
                    torch.exp(self.params['log_scales']),
                    dim=1).values <= 0.01 * self.variables['scene_radius']))
                new_params = {
                    k: v[to_clone]
                    for k, v in self.params.items() if 'pose' not in k
                }
                self.cat_params_to_optimizer(new_params, optimizer)
                num_pts = self.params['means3D'].shape[0]

                padded_grad = torch.zeros(num_pts, device='cuda')
                padded_grad[:grads.shape[0]] = grads
                to_split = torch.logical_and(
                    padded_grad >= grad_thresh,
                    torch.max(torch.exp(self.params['log_scales']),
                              dim=1).values >
                    0.01 * self.variables['scene_radius'])
                n = self.densify_dict[
                    'num_to_split_into']  # number to split into
                new_params = {
                    k: v[to_split].repeat(n, 1)
                    for k, v in self.params.items() if 'pose' not in k
                }
                stds = torch.exp(self.params['log_scales'])[to_split].repeat(
                    n, 3)
                means = torch.zeros((stds.size(0), 3), device='cuda')
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(
                    self.params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
                new_params['means3D'] += torch.bmm(
                    rots, samples.unsqueeze(-1)).squeeze(-1)
                new_params['log_scales'] = torch.log(
                    torch.exp(new_params['log_scales']) / (0.8 * n))
                self.cat_params_to_optimizer(new_params, optimizer)
                num_pts = self.params['means3D'].shape[0]

                self.variables['means2D_gradient_accum'] = torch.zeros(
                    num_pts, device='cuda')
                self.variables['denom'] = torch.zeros(num_pts, device='cuda')
                self.variables['max_2D_radius'] = torch.zeros(num_pts,
                                                              device='cuda')
                to_remove = torch.cat((to_split,
                                       torch.zeros(n * to_split.sum(),
                                                   dtype=torch.bool,
                                                   device='cuda')))
                self.remove_points(to_remove, optimizer)

                if iter == self.densify_dict['stop_after']:
                    remove_threshold = self.densify_dict[
                        'final_removal_opacity_threshold']
                else:
                    remove_threshold = self.densify_dict[
                        'removal_opacity_threshold']
                to_remove = (torch.sigmoid(self.params['logit_opacities']) <
                             remove_threshold).squeeze()
                if iter >= self.densify_dict['remove_big_after']:
                    big_points_ws = torch.exp(self.params['log_scales']).max(
                        dim=1).values > 0.1 * self.variables['scene_radius']
                    to_remove = torch.logical_or(to_remove, big_points_ws)
                self.remove_points(to_remove, optimizer)

                torch.cuda.empty_cache()

            # Reset Opacities for all Gaussians (This is not desired for
            # mapping on only current frame)
            if iter > 0 and iter % self.densify_dict[
                    'reset_opacities_every'] == 0 and self.densify_dict[
                        'reset_opacities']:
                new_params = {
                    'logit_opacities':
                    inverse_sigmoid(
                        torch.ones_like(self.params['logit_opacities']) * 0.01)
                }
                self.update_params_and_optimizer(new_params, optimizer)

    def add_new_gaussians(self, gt_rgb, gt_depth, curr_w2c, sil_thres,
                          mean_sq_dist_method):
        # Silhouette Rendering
        transformed_pts = transform_to_frame(self.params['means3D'],
                                             curr_w2c,
                                             gaussians_grad=False,
                                             camera_grad=False)
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            self.params, self.first_frame_w2c, transformed_pts)
        depth_sil, _, _, = Renderer(raster_settings=self.gaussian_cam)(
            **depth_sil_rendervar)
        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)

        # Check for new foreground objects by using GT depth
        render_depth = depth_sil[0, :, :]
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        non_presence_depth_mask = (render_depth > gt_depth) * (
            depth_error > 50 * depth_error.median())
        # Determine non-presence mask
        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
        # Flatten mask
        non_presence_mask = non_presence_mask.reshape(-1)

        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # Get the new pointcloud in the world frame
            valid_depth_mask = (gt_depth > 0)
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(
                -1)
            new_pt_cld, mean3_sq_dist = self.get_pointcloud(
                color=gt_rgb,
                depth=gt_depth,
                w2c=curr_w2c.detach(),
                mask=non_presence_mask,
                compute_mean_sq_dist=True,
                mean_sq_dist_method=mean_sq_dist_method)
            new_params, _ = self.initialize_params(new_pt_cld, mean3_sq_dist)
            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(
                    torch.cat((self.params[k], v), dim=0).requires_grad_(True))
            num_pts = self.params['means3D'].shape[0]
            self.variables['means2D_gradient_accum'] = torch.zeros(
                num_pts, device='cuda').float()
            self.variables['denom'] = torch.zeros(num_pts,
                                                  device='cuda').float()
            self.variables['max_2D_radius'] = torch.zeros(
                num_pts, device='cuda').float()

    def initialize_params(self, pt_cld, mean3_sq_dist):
        num_pts = pt_cld.shape[0]
        means3D = pt_cld[:, :3]  # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
        logit_opacities = torch.zeros((num_pts, 1),
                                      dtype=torch.float,
                                      device='cuda')
        params = {
            'means3D':
            means3D,
            'rgb_colors':
            pt_cld[:, 3:6],
            'unnorm_rotations':
            unnorm_rots,
            'logit_opacities':
            logit_opacities,
            'log_scales':
            torch.tile(
                torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        }

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(
                    torch.tensor(v).cuda().float().contiguous().requires_grad_(
                        True))
            else:
                params[k] = torch.nn.Parameter(
                    v.cuda().float().contiguous().requires_grad_(True))

        variables = {
            'max_2D_radius':
            torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'means2D_gradient_accum':
            torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'denom':
            torch.zeros(params['means3D'].shape[0]).cuda().float()
        }  # 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()
        return params, variables

    def get_pointcloud(self,
                       color,
                       depth,
                       w2c,
                       mask=None,
                       compute_mean_sq_dist=True,
                       mean_sq_dist_method='projective'):
        width, height = self.camera.width, self.camera.height
        CX = self.camera.cx
        CY = self.camera.cy
        FX = self.camera.fx
        FY = self.camera.fy
        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX) / FX
        yy = (y_grid - CY) / FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)
        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
        # Compute mean squared distance for initializing the scale of
        # the Gaussians
        if compute_mean_sq_dist:
            if mean_sq_dist_method == 'projective':
                # Projective Geometry (this is fast, farther -> larger radius)
                scale_gaussian = depth_z / ((FX + FY) / 2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(
                    f'Unknown mean_sq_dist_method {mean_sq_dist_method}')

        # Colorize point cloud
        cols = color.reshape(-1, 3)  # (H, W, C) -> (H * W, C)
        point_cld = torch.cat((pts, cols), -1)

        # Select points based on mask
        if mask is not None:
            point_cld = point_cld[mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return point_cld, mean3_sq_dist
        else:
            return point_cld, None
