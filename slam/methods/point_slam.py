import functools
from dataclasses import dataclass, field
from typing import Type

import cv2
import numpy as np
import torch
import trimesh
from scipy.interpolate import interp1d
from skimage import filters
from skimage.color import rgb2gray

from slam.common.camera import Camera
from slam.common.common import (clean_mesh, get_mesh_from_RGBD, get_rays,
                                get_samples, get_samples_with_pixel_grad)
from slam.methods.base_method import Method, MethodConfig
from slam.models.conv_onet2 import ConvOnet2Config


@dataclass
class PointSLAMConfig(MethodConfig):
    """PointSLAM  Config."""
    _target: Type = field(default_factory=lambda: PointSLAM)

    use_dynamic_radius: bool = True
    pixels_adding: int = 6000

    # sample
    mapping_sample: int = 2048
    min_sample_pixels: int = 100
    tracking_sample: int = 1024
    # render image
    ray_batch_size: int = 3000

    # tracking
    tracking_sample_with_color_grad: bool = False
    tracking_Wedge: int = 100
    tracking_Hedge: int = 100

    # mapping
    mapping_geo_iter_ratio: float = 0.4
    mapping_pixels_based_on_color_grad: int = 0
    mapping_frustum_feature_selection: bool = True
    mapping_frustum_edge: int = -4
    mapping_BA: bool = False

    # model
    model_encode_exposure: bool = False

    # pointcloud
    pointcloud_radius_add_max: float = 0.08
    pointcloud_radius_add_min: float = 0.02
    pointcloud_radius_add: float = 0.04
    pointcloud_radius_query: float = 0.08
    pointcloud_radius_query_ratio: int = 2
    pointcloud_color_grad_threshold: float = 0.15

    # mesh
    clean_mesh: bool = True


class PointSLAM(Method):

    config: PointSLAMConfig
    model_config: ConvOnet2Config

    def __init__(self, config: PointSLAMConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.stage = 'color'
        # set model config
        model_config = self.config.model
        model_config.model_encode_exposure = self.config.model_encode_exposure
        model_config.use_dynamic_radius = self.config.use_dynamic_radius
        model_config.mapping_pixels_based_on_color_grad = \
            self.config.mapping_pixels_based_on_color_grad
        self.model = model_config.setup(camera=camera)
        self.model.to(device)
        self.dynamic_r_query_allkeyframe = {}

    def pre_precessing(self, cur_frame, is_mapping):
        gt_depth_np = cur_frame.depth
        gt_color_np = cur_frame.rgb
        c2w = cur_frame.get_pose()
        idx = cur_frame.fid

        batch_dynamic_r = None
        if self.config.use_dynamic_radius:
            dynamic_r_add, dynamic_r_query = self.cal_dynamic_radius(
                gt_color_np)
            self.dynamic_r_query_allkeyframe[np.array2string(
                idx)] = dynamic_r_query

        if is_mapping:
            # use curframe data to update neural_point_cloud
            gt_depth = torch.from_numpy(gt_depth_np).to(self.device)
            if idx == 0:
                add_pts_num = torch.clamp(
                    self.config.pixels_adding * ((gt_depth.median() / 2.5)**2),
                    min=self.config.pixels_adding,
                    max=self.config.pixels_adding * 3).int().item()
            else:
                add_pts_num = self.config.pixels_adding
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, \
                i, j = get_samples(self.camera,
                                   add_pts_num,
                                   c2w,
                                   gt_depth_np,
                                   gt_color_np,
                                   device=self.device,
                                   depth_filter=True,
                                   return_index=True)
            batch_dynamic_r = None
            if self.config.use_dynamic_radius:
                batch_dynamic_r = dynamic_r_add[j, i]
            input = {
                'batch_rays_o': batch_rays_o,
                'batch_rays_d': batch_rays_d,
                'batch_gt_depth': batch_gt_depth,
                'batch_gt_color': batch_gt_color,
                'batch_dynamic_r': batch_dynamic_r
            }
            batch_dynamic_r_grad = None
            if self.config.mapping_pixels_based_on_color_grad > 0:
                batch_rays_o_grad, batch_rays_d_grad, batch_gt_depth_grad, \
                    batch_gt_color_grad, i_grad, j_grad = \
                    get_samples_with_pixel_grad(
                            self.camera,
                            self.config.mapping_pixels_based_on_color_grad,
                            c2w,
                            gt_depth_np,
                            gt_color_np,
                            device=self.device,
                            depth_filter=True,
                            return_index=True
                            )
                if self.config.use_dynamic_radius:
                    batch_dynamic_r_grad = dynamic_r_add[j_grad, i_grad]

                input = {
                    **input, 'batch_rays_o_grad': batch_rays_o_grad,
                    'batch_rays_d_grad': batch_rays_d_grad,
                    'batch_gt_depth_grad': batch_gt_depth_grad,
                    'batch_gt_color_grad': batch_gt_color_grad,
                    'batch_dynamic_r_grad': batch_dynamic_r_grad
                }

            self.model.model_update(input)
            # set masked_indices
            masked_indices = None
            if self.config.mapping_frustum_feature_selection:
                mask_c2w = c2w
                masked_indices = self.get_mask_from_c2w(mask_c2w, gt_depth_np)
                self.model.masked_indices = torch.tensor(masked_indices,
                                                         device=self.device)

    def optimizer_config_update(self, max_iters, coarse=False):
        self.bundle_adjust = (len(self.keyframe_graph) >
                              4) and self.config.mapping_BA
        for _, params in self.config.optimizers.items():
            if params['scheduler'] is not None:
                params['optimizer'].lr = 1.0
                params['scheduler'].max_steps = max_iters
                params['scheduler'].geo_iter_ratio = \
                    self.config.mapping_geo_iter_ratio

    def get_model_input(self, optimize_frames, is_mapping):
        batch_rays_d_list = []
        batch_rays_o_list = []
        batch_gt_depth_list = []
        batch_gt_color_list = []
        batch_r_query_list = []

        pixs_per_image = self.config.tracking_sample
        Hedge = self.config.tracking_Hedge
        Wedge = self.config.tracking_Wedge
        if is_mapping:
            pixs_per_image = np.maximum(
                self.config.mapping_sample // len(optimize_frames),
                self.config.min_sample_pixels)
            Hedge = 0
            Wedge = 0
        for frame in optimize_frames:
            gt_depth = frame.depth
            gt_color = frame.rgb
            c2w = frame.get_pose()
            if self.config.use_dynamic_radius:
                dynamic_r_query = self.dynamic_r_query_allkeyframe[
                    np.array2string(frame.fid)]
            if not is_mapping and self.config.tracking_sample_with_color_grad:
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, \
                    i, j = get_samples_with_pixel_grad(
                        self.camera,
                        pixs_per_image,
                        c2w,
                        gt_depth,
                        gt_color,
                        device=self.device,
                        Hedge=Hedge,
                        Wedge=Wedge,
                        depth_filter=True,
                        return_index=True)
            else:
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, \
                    i, j = get_samples(self.camera,
                                       pixs_per_image,
                                       c2w,
                                       gt_depth,
                                       gt_color,
                                       device=self.device,
                                       Hedge=Hedge,
                                       Wedge=Wedge,
                                       depth_filter=True,
                                       return_index=True)
            batch_rays_o_list.append(batch_rays_o.float())
            batch_rays_d_list.append(batch_rays_d.float())
            batch_gt_depth_list.append(batch_gt_depth.float())
            batch_gt_color_list.append(batch_gt_color.float())
            if self.config.use_dynamic_radius:
                batch_r_query_list.append(dynamic_r_query[j, i])

        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)
        r_query_list = torch.cat(
            batch_r_query_list) if self.config.use_dynamic_radius else None

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            inside_mask = batch_gt_depth <= torch.minimum(
                10 * batch_gt_depth.median(), 1.2 * torch.max(batch_gt_depth))

        batch_rays_d = batch_rays_d[inside_mask]  # (N, 3)
        batch_rays_o = batch_rays_o[inside_mask]  # (N, 3)
        batch_gt_depth = batch_gt_depth[inside_mask]  # (N, 1)
        batch_gt_color = batch_gt_color[inside_mask]  # (N, 3)
        batch_dynamic_r = None
        if self.config.use_dynamic_radius:
            batch_dynamic_r = r_query_list[inside_mask]

        return {
            'rays_o': batch_rays_o,
            'rays_d': batch_rays_d,
            'target_s': batch_gt_color,
            'target_d': batch_gt_depth,
            'batch_dynamic_r': batch_dynamic_r,
            'stage': self.stage,
        }

    def set_stage(self, is_mapping, step, n_iters):
        if not is_mapping:
            self.stage = 'color'
            return
        if step <= int(n_iters * self.config.mapping_geo_iter_ratio):
            self.stage = 'geometry'
        else:
            self.stage = 'color'

    def get_loss(self,
                 optimize_frames,
                 is_mapping,
                 step,
                 n_iters,
                 coarse=False):
        self.set_stage(is_mapping, step, n_iters)
        model_input = self.get_model_input(optimize_frames, is_mapping)
        model_outputs = self.model(model_input)
        loss_dict = self.model.get_loss_dict(model_outputs, model_input,
                                             is_mapping, self.stage)
        loss = functools.reduce(torch.add, loss_dict.values())
        return loss

    def render_img(self, c2w, gt_depth=None, idx=None):
        with self.lock and torch.no_grad():
            rays_o, rays_d = get_rays(self.camera, c2w, device=self.device)

            rays_o = rays_o.reshape(-1, 3)  # (H, W, 3)->(H*W, 3)
            rays_d = rays_d.reshape(-1, 3)

            if self.config.use_dynamic_radius:
                dynamic_r_query = self.dynamic_r_query_allkeyframe[
                    np.array2string(idx)].reshape(-1, 1)

            depth_list = []
            uncertainty_list = []
            color_list = []

            if gt_depth is not None:
                gt_depth = torch.from_numpy(gt_depth).to(
                    self.device).reshape(-1)

            # run batch by batch
            ray_batch_size = self.config.ray_batch_size
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]

                model_input_batch = {
                    'rays_o': rays_o_batch,
                    'rays_d': rays_d_batch,
                    'target_s': None,
                    'target_d': None,
                    'stage': 'color',
                    'batch_dynamic_r': None,
                }

                if self.config.use_dynamic_radius:
                    model_input_batch['batch_dynamic_r'] = dynamic_r_query[
                        i:i + ray_batch_size]

                if gt_depth is not None:
                    model_input_batch['target_d'] = gt_depth[i:i +
                                                             ray_batch_size]

                model_outputs = self.model(model_input_batch)
                depth_list.append(model_outputs['depth'].double())
                color_list.append(model_outputs['rgb'])
                uncertainty_list.append(model_outputs['uncertainty'].double())

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(self.camera.height, self.camera.width)
            uncertainty = uncertainty.reshape(self.camera.height,
                                              self.camera.width)
            color = color.reshape(self.camera.height, self.camera.width, 3)

            return color.clone().squeeze().cpu().numpy(), depth.clone(
            ).squeeze().cpu().numpy()

    def get_cloud(self, c2w_np, gt_depth_np):
        with self.lock and torch.no_grad():
            cloud_pos = np.array(self.model.neural_point_cloud.input_pos())
            cloud_rgb = np.array(self.model.neural_point_cloud.input_rgb())
            return cloud_pos, cloud_rgb / 255.0

    def cal_dynamic_radius(self, gt_color_np):
        ratio = self.config.pointcloud_radius_query_ratio
        intensity = rgb2gray(gt_color_np)
        grad_y = filters.sobel_h(intensity)
        grad_x = filters.sobel_v(intensity)
        color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
        color_grad_mag = np.clip(
            color_grad_mag, 0.0,
            self.config.pointcloud_color_grad_threshold)  # range 0~1
        fn_map_r_add = interp1d(
            [0, 0.01, self.config.pointcloud_color_grad_threshold], [
                self.config.pointcloud_radius_add_max,
                self.config.pointcloud_radius_add_max,
                self.config.pointcloud_radius_add_min
            ])
        fn_map_r_query = interp1d(
            [0, 0.01, self.config.pointcloud_color_grad_threshold], [
                ratio * self.config.pointcloud_radius_add_max,
                ratio * self.config.pointcloud_radius_add_max,
                ratio * self.config.pointcloud_radius_add_min
            ])
        dynamic_r_add = fn_map_r_add(color_grad_mag)
        dynamic_r_query = fn_map_r_query(color_grad_mag)
        dynamic_r_add, dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
            self.device), torch.from_numpy(dynamic_r_query).to(self.device)
        return dynamic_r_add, dynamic_r_query

    def get_mask_from_c2w(self, c2w, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
        """
        H, W, fx, fy, cx, cy, = (self.camera.height, self.camera.width,
                                 self.camera.fx, self.camera.fy,
                                 self.camera.cx, self.camera.cy)
        points = np.array(self.model.neural_point_cloud._cloud_pos).reshape(
            -1, 3)

        c2w = c2w.detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([points, ones],
                                       axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # flip the x-axis such that the pixel space is u from the left to
        # right, v top to bottom.
        # without the flipping of the x-axis, the image is assumed to be
        # flipped horizontally.
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [
                cv2.remap(depth_np,
                          uv[i:i + remap_chunk, 0],
                          uv[i:i + remap_chunk, 1],
                          interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)
            ]
        depths = np.concatenate(depths, axis=0)

        edge = self.config.mapping_frustum_edge
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
        mask = mask.reshape(-1)
        return mask

    def get_mesh(self):
        with self.lock and torch.no_grad():
            o3d_mesh = get_mesh_from_RGBD(self.camera, self.keyframe_graph)
            if self.config.clean_mesh:
                self.cur_mesh = clean_mesh(o3d_mesh)
            else:
                vertices = o3d_mesh.vertices
                faces = o3d_mesh.triangles
                self.cur_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return self.cur_mesh
