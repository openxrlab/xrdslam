from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.model_components.decoder_pointslam import POINT
from slam.model_components.neural_point_cloud import NeuralPointCloud
from slam.model_components.utils import raw2outputs_nerf_color2
from slam.models.base_model import Model, ModelConfig


@dataclass
class ConvOnet2Config(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: ConvOnet2)

    use_dynamic_radius: bool = True
    points_batch_size: int = 50000
    cuda_id: int = 0

    # pretrain
    pretrained_decoders_middle_fine: Optional[Path] = None

    # model
    model_c_dim: int = 32
    model_pos_embedding_method: str = 'fourier'
    model_use_view_direction: bool = False
    model_encode_rel_pos_in_col: bool = True
    model_encode_exposure: bool = False
    model_encode_viewd: bool = True
    model_exposure_dim: int = 8

    # pointcloud
    # 'distance'|'expo" whether to use e(-x) or inverse square distance
    # for weighting
    pointcloud_nn_weighting: str = 'distance'
    # how many nn to choose at most within search radius
    pointcloud_nn_num: int = 8
    # if nn_num less than this, will skip this sample location
    pointcloud_min_nn_num: int = 2
    pointcloud_radius_add: float = 0.04
    pointcloud_radius_min: float = 0.02
    pointcloud_radius_query: float = 0.08
    pointcloud_fix_interval_when_add_along_ray: bool = False
    pointcloud_n_add: int = 3

    # rendering
    rendering_n_surface: int = 5
    rendering_sample_near_pcl: bool = False
    rendering_near_end_surface: float = 0.98
    rendering_near_end: float = 0.3
    rendering_far_end_surface: float = 1.02
    rendering_sigmoid_coef_mapper: float = 0.1

    # color loss weight
    tracking_w_color_loss: float = 0.5
    mapping_w_color_loss: float = 0.1

    # tracking
    tracking_handle_dynamic: bool = True
    tracking_use_color_in_tracking: bool = True

    # mapping
    mapping_fix_color_decoder: bool = False
    mapping_fix_geo_decoder: bool = True
    mapping_pixels_based_on_color_grad: int = 1000


class ConvOnet2(Model):
    """Model class."""

    config: ConvOnet2Config

    def __init__(
        self,
        config: ConvOnet2Config,
        camera: Camera,
        **kwargs,
    ) -> None:
        super().__init__(config=config,
                         camera=camera,
                         bounding_box=None,
                         **kwargs)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()
        self.get_decoder()
        self.load_pretrain()
        self.masked_indices = None
        self.neural_point_cloud = None

    def model_update(self, input):
        if self.neural_point_cloud is None:
            self.neural_point_cloud = NeuralPointCloud(
                c_dim=self.config.model_c_dim,
                cuda_id=self.config.cuda_id,
                nn_num=self.config.pointcloud_nn_num,
                radius_add=self.config.pointcloud_radius_add,
                radius_min=self.config.pointcloud_radius_min,
                radius_query=self.config.pointcloud_radius_query,
                fix_interval_when_add_along_ray=self.config.
                pointcloud_fix_interval_when_add_along_ray,
                use_dynamic_radius=self.config.use_dynamic_radius,
                N_surface=self.config.rendering_n_surface,
                N_add=self.config.pointcloud_n_add,
                near_end_surface=self.config.rendering_near_end_surface,
                far_end_surface=self.config.rendering_far_end_surface,
                device=self.device)
        self.neural_point_cloud.add_neural_points(
            batch_rays_o=input['batch_rays_o'],
            batch_rays_d=input['batch_rays_d'],
            batch_gt_depth=input['batch_gt_depth'],
            batch_gt_color=input['batch_gt_color'],
            dynamic_radius=input['batch_dynamic_r'])
        if self.config.mapping_pixels_based_on_color_grad > 0:
            self.neural_point_cloud.add_neural_points(
                batch_rays_o=input['batch_rays_o_grad'],
                batch_rays_d=input['batch_rays_d_grad'],
                batch_gt_depth=input['batch_gt_depth_grad'],
                batch_gt_color=input['batch_gt_color_grad'],
                dynamic_radius=input['batch_dynamic_r_grad'],
                is_pts_grad=True)

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        rays_o = input['rays_o']
        rays_d = input['rays_d']
        target_d = input['target_d']
        stage = input['stage']
        dynamic_r_query = input['batch_dynamic_r']
        outputs = self.render_batch_ray(rays_d=rays_d,
                                        rays_o=rays_o,
                                        stage=stage,
                                        gt_depth=target_d,
                                        dynamic_r_query=dynamic_r_query)
        outputs['stage'] = stage
        return outputs

    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        target_d = inputs['target_d'].squeeze()
        target_rgb = inputs['target_s']
        depth = outputs['depth']
        color = outputs['rgb']
        uncertainty = outputs['uncertainty']
        valid_ray_mask = outputs['valid_ray_mask']
        stage = outputs['stage']
        # tracking loss
        if not is_mapping:
            w_color_loss = self.config.tracking_w_color_loss
            uncertainty = uncertainty.detach()
            nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
            if self.config.tracking_handle_dynamic:
                tmp = torch.abs(target_d - depth) / torch.sqrt(uncertainty +
                                                               1e-10)
                mask = (tmp < 10 * tmp.median()) & (target_d > 0)
            else:
                tmp = torch.abs(target_d - depth)
                mask = (tmp < 10 * tmp.median()) & (target_d > 0)
            mask = mask & nan_mask
            geo_loss = torch.clamp((torch.abs(target_d - depth) /
                                    torch.sqrt(uncertainty + 1e-10)),
                                   min=0.0,
                                   max=1e3)[mask].sum()
            loss_dict['geo_loss'] = geo_loss
            if self.config.tracking_use_color_in_tracking:
                color_loss = torch.abs(target_rgb - color)[mask].sum()
                weighted_color_loss = w_color_loss * color_loss
                loss_dict['rgb_loss'] = weighted_color_loss
        # mapping loss
        else:
            w_color_loss = self.config.mapping_w_color_loss
            depth_mask = (target_d > 0) & valid_ray_mask
            depth_mask = depth_mask & (~torch.isnan(depth))
            geo_loss = torch.abs(target_d[depth_mask] -
                                 depth[depth_mask]).sum()
            loss_dict['geo_loss'] = geo_loss
            if stage == 'color':
                if self.config.model_encode_exposure:
                    # TODO: support encode exposure
                    pass
                color_loss = torch.abs(target_rgb[depth_mask] -
                                       color[depth_mask]).sum()
                weighted_color_loss = w_color_loss * color_loss
                loss_dict['rgb_loss'] = weighted_color_loss
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        decoders_para_list = []
        if not self.config.mapping_fix_geo_decoder:
            decoders_para_list += list(self.decoder.geo_decoder.parameters())
        if not self.config.mapping_fix_color_decoder:
            decoders_para_list += list(self.decoder.color_decoder.parameters())
        param_groups['decoder'] = decoders_para_list
        # neural_point_cloud params
        if self.masked_indices is not None:
            self.neural_point_cloud.set_mask(self.masked_indices)
        param_groups['geometry'] = [self.neural_point_cloud.geo_feats]
        param_groups['color'] = [self.neural_point_cloud.col_feats]
        return param_groups

    def get_decoder(self):
        """Get the decoder of the scene representation."""
        self.decoder = POINT(
            use_dynamic_radius=self.config.use_dynamic_radius,
            pointcloud_nn_weighting=self.config.pointcloud_nn_weighting,
            pointcloud_min_nn_num=self.config.pointcloud_min_nn_num,
            rendering_n_surface=self.config.rendering_n_surface,
            model_encode_rel_pos_in_col=self.config.
            model_encode_rel_pos_in_col,
            model_encode_exposure=self.config.model_encode_exposure,
            model_encode_viewd=self.config.model_encode_viewd,
            model_exposure_dim=self.config.model_exposure_dim,
            c_dim=self.config.model_c_dim,
            pos_embedding_method=self.config.model_pos_embedding_method,
            use_view_direction=self.config.model_use_view_direction)

    def load_pretrain(self):
        """Load parameters of pretrained ConvOnet checkpoints to the
        decoders."""
        ckpt = torch.load(self.config.pretrained_decoders_middle_fine,
                          map_location=self.device)
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8 + 7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8 + 5:]
                    fine_dict[key] = val
        self.decoder.geo_decoder.load_state_dict(middle_dict, strict=False)

    def eval_points(self,
                    p,
                    stage='color',
                    is_tracker=False,
                    pts_views_d=None,
                    ray_pts_num=None,
                    dynamic_r_query=None,
                    exposure_feat=None):
        """Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            stage (str, optional): 'geometry'|'color', defaults to 'color'.
            device (str, optional): CUDA device.
            is_tracker (bool, optional): tracker has different gradient
            flow in eval_points.
            cloud_pos (tensor, optional): positions of all point cloud
            features, used only when tracker calls.
            pts_views_d (tensor, optional): ray direction for each point
            ray_pts_num (tensor, optional): number of surface samples
            dynamic_r_query (tensor, optional): if use dynamic query,
            for every ray, its query radius is different.
            exposure_feat (tensor, optional): whether to render with
            an exposure feature vector. All rays have the same
            exposure feature vector.

        Returns:
            ret (tensor): occupancy (and color) value of input points, (N,)
            valid_ray_mask (tensor):
        """
        assert torch.is_tensor(p)
        p_split = torch.split(p, self.config.points_batch_size)
        rets = []
        ray_masks = []
        point_masks = []
        for pi in p_split:
            pi = pi.unsqueeze(0)
            ret, valid_ray_mask, point_mask = self.decoder(
                p=pi,
                npc=self.neural_point_cloud,
                stage=stage,
                pts_num=ray_pts_num,
                is_tracker=is_tracker,
                pts_views_d=pts_views_d,
                dynamic_r_query=dynamic_r_query,
                exposure_feat=exposure_feat)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            rets.append(ret)
            ray_masks.append(valid_ray_mask)
            point_masks.append(point_mask)

        ret = torch.cat(rets, dim=0)
        ray_mask = torch.cat(ray_masks, dim=0)
        point_mask = torch.cat(point_masks, dim=0)

        return ret, ray_mask, point_mask

    def render_batch_ray(self,
                         rays_d,
                         rays_o,
                         stage,
                         gt_depth=None,
                         is_tracker=True,
                         dynamic_r_query=None,
                         exposure_feat=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]
            stage (str): query stage.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty
            (can be interpreted as epistemic uncertainty)
            color (tensor): rendered color.
            valid_ray_mask (tensor): filter corner cases.
        '''
        N_rays = rays_o.shape[0]
        if gt_depth is not None:
            # per ray far rendering distance for pixels that have no depth
            # if the max depth is an outlier, it will be very large.
            # Use 5*mean depth instead then.
            far = torch.minimum(5 * gt_depth.mean(),
                                torch.max(gt_depth * 1.2)).repeat(
                                    rays_o.shape[0], 1).float()

            if torch.numel(gt_depth) != 0:
                gt_depth = gt_depth.reshape(-1, 1)
            else:
                # handle error, gt_depth is empty
                print('tensor gt_depth is empty, info:')
                gt_depth = torch.zeros(N_rays, 1, device=self.device)
        else:
            # render over 10 m when no depth is available at all
            far = 10 * \
                torch.ones((rays_o.shape[0], 1), device=self.device).float()
            gt_depth = torch.zeros(N_rays, 1, device=self.device)

        gt_non_zero_mask = gt_depth > 0
        gt_non_zero_mask = gt_non_zero_mask.squeeze(-1)
        mask_rays_near_pcl = torch.ones(N_rays, device=self.device).type(
            torch.bool).to(self.device)

        gt_non_zero = gt_depth[gt_non_zero_mask]
        gt_depth_surface = gt_non_zero.repeat(1,
                                              self.config.rendering_n_surface)

        t_vals_surface = torch.linspace(0.0,
                                        1.0,
                                        steps=self.config.rendering_n_surface,
                                        device=self.device)

        z_vals_surface_depth_none_zero = \
            self.config.rendering_near_end_surface * gt_depth_surface * \
            (1.-t_vals_surface) + self.config.rendering_far_end_surface * \
            gt_depth_surface * (t_vals_surface)

        z_vals_surface = torch.zeros(gt_depth.shape[0],
                                     self.config.rendering_n_surface,
                                     device=self.device)
        z_vals_surface[gt_non_zero_mask, :] = z_vals_surface_depth_none_zero

        if gt_non_zero_mask.sum() < N_rays:
            # determine z_vals_surface values for zero-valued depth pixels
            if self.config.rendering_sample_near_pcl:
                # do ray marching from near_end to far, check if there is
                # a line segment close to point cloud
                # we sample 25 points between near_end and far_end
                # the mask_not_near_pcl is True for rays that are not close
                # to the npc
                z_vals_depth_zero, mask_not_near_pcl = \
                    self.neural_point_cloud.sample_near_pcl(
                        rays_o[~gt_non_zero_mask].detach().clone(),
                        rays_d[~gt_non_zero_mask].detach().clone(),
                        self.config.rendering_near_end, torch.max(far),
                        self.config.rendering_n_surface)
                if torch.sum(mask_not_near_pcl.ravel()):
                    # after ray marching, some rays are not close to
                    # the point cloud
                    rays_not_near = torch.nonzero(
                        ~gt_non_zero_mask, as_tuple=True)[0][mask_not_near_pcl]
                    # update the mask_rays_near_pcl to False for the rays where
                    # mask_not_near_pcl is True
                    mask_rays_near_pcl[rays_not_near] = False
                z_vals_surface[~gt_non_zero_mask, :] = z_vals_depth_zero
            else:
                # simply sample uniformly
                z_vals_surface[~gt_non_zero_mask, :] = torch.linspace(
                    self.config.rendering_near_end,
                    torch.max(far),
                    steps=self.config.rendering_n_surface,
                    device=self.device).repeat((~gt_non_zero_mask).sum(), 1)

        z_vals = z_vals_surface

        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pointsf = pts.reshape(-1, 3)

        ray_pts_num = self.config.rendering_n_surface
        rays_d_pts = rays_d.repeat_interleave(ray_pts_num,
                                              dim=0).reshape(-1, 3)
        if self.config.use_dynamic_radius:
            dynamic_r_query = dynamic_r_query.reshape(-1, 1).repeat_interleave(
                ray_pts_num, dim=0)

        raw, valid_ray_mask, point_mask = self.eval_points(
            p=pointsf,
            stage=stage,
            is_tracker=is_tracker,
            pts_views_d=rays_d_pts,
            ray_pts_num=ray_pts_num,
            dynamic_r_query=dynamic_r_query,
            exposure_feat=exposure_feat)

        with torch.no_grad():
            raw[torch.nonzero(~point_mask).flatten(), -1] = -100.0
        raw = raw.reshape(N_rays, ray_pts_num, -1).to(self.device)
        depth, uncertainty, color, weights = raw2outputs_nerf_color2(
            raw,
            z_vals,
            rays_d,
            device=self.device,
            coef=self.config.rendering_sigmoid_coef_mapper)

        # filter two cases:
        # 1. ray has no gt_depth and it's not close to the current npc
        # 2. ray has gt_depth, but all its sampling locations have no
        # neighbors in current npc
        valid_ray_mask = valid_ray_mask.to(self.device)
        valid_ray_mask = valid_ray_mask & mask_rays_near_pcl

        if not self.config.rendering_sample_near_pcl:
            depth[~gt_non_zero_mask] = 0

        # Return rendering outputs
        ret = {
            'rgb': color,
            'depth': depth,
            'uncertainty': uncertainty,
            'valid_ray_mask': valid_ray_mask
        }

        return ret
