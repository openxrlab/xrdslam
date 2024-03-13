from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.model_components.decoder_nice import NICE
from slam.model_components.feature_grid_nice import FeatureGrid
from slam.model_components.utils import (get_mask_from_c2w,
                                         raw2outputs_nerf_color, sample_pdf)
from slam.models.base_model import Model, ModelConfig


@dataclass
class ConvOnetConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: ConvOnet)

    coarse: bool = False  # TODO: support True
    occupancy: bool = True

    # pretrain
    pretrained_decoders_coarse: Optional[Path] = None
    pretrained_decoders_middle_fine: Optional[Path] = None

    # data
    data_dim: int = 3

    # model
    model_c_dim: int = 32
    model_pos_embedding_method: str = 'fourier'
    model_coarse_bound_enlarge: int = 2

    # grid
    grid_len_coarse: float = 2
    grid_len_middle: float = 0.32
    grid_len_fine: float = 0.16
    grid_len_color: float = 0.16
    grid_bound_divisible: float = 0.32

    # rendering
    rendering_n_samples: int = 32
    rendering_n_surface: int = 16
    rendering_n_importance: int = 0
    rendering_lindisp: bool = False
    rendering_perturb: float = 0.0
    points_batch_size: int = 500000

    # training
    tracking_w_color_loss: float = 0.5
    mapping_w_color_loss: float = 0.2

    # tracking
    tracking_handle_dynamic: bool = True
    tracking_use_color_in_tracking: bool = True

    # mapping
    mapping_fix_fine: bool = True
    mapping_fix_color: bool = False
    mapping_frustum_feature_selection: bool = True


class ConvOnet(Model):
    """Model class."""

    config: ConvOnetConfig

    def __init__(
        self,
        config: ConvOnetConfig,
        camera: Camera,
        bounding_box,
        **kwargs,
    ) -> None:
        super().__init__(config=config,
                         camera=camera,
                         bounding_box=bounding_box,
                         **kwargs)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()
        self.get_decoder()
        self.load_bound()
        self.load_pretrain()
        self.grid_init()
        self.grid_opti_mask = {}

    def pre_precessing(self, cur_frame):
        if self.config.mapping_frustum_feature_selection:
            gt_depth_np = cur_frame.depth
            c2w = cur_frame.get_pose()
            for key, grid in self.grid_c.items():
                if key != 'grid_coarse':
                    mask = get_mask_from_c2w(camera=self.camera,
                                             bound=self.bounding_box,
                                             c2w=c2w,
                                             key=key,
                                             val_shape=grid.val.shape[2:],
                                             depth_np=gt_depth_np)
                    mask = torch.from_numpy(mask).permute(
                        2, 1, 0).unsqueeze(0).unsqueeze(0).repeat(
                            1, grid.val.shape[1], 1, 1, 1)
                    self.grid_opti_mask[key] = mask

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        rays_o = input['rays_o']
        rays_d = input['rays_d']
        target_d = input['target_d']
        stage = input['stage']
        if stage == 'coarse':
            target_d = None
        outputs = self.render_batch_ray(rays_o,
                                        rays_d,
                                        stage,
                                        gt_depth=target_d)
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
        uncertainty = uncertainty.detach()
        # tracking loss
        if not is_mapping:
            w_color_loss = self.config.tracking_w_color_loss
            if self.config.tracking_handle_dynamic:
                tmp = torch.abs(target_d - depth) / torch.sqrt(uncertainty +
                                                               1e-10)
                depth_mask = (tmp < 10 * tmp.median()) & (target_d > 0)
            else:
                depth_mask = target_d > 0
            depth_loss = (torch.abs(target_d - depth) /
                          torch.sqrt(uncertainty + 1e-10))[depth_mask].sum()
            loss_dict['depth_loss'] = depth_loss
            if self.config.tracking_use_color_in_tracking:
                color_loss = torch.abs(target_rgb - color)[depth_mask].sum()
                weighted_color_loss = w_color_loss * color_loss
                loss_dict['rgb_loss'] = weighted_color_loss
        # mapping loss
        else:
            w_color_loss = self.config.mapping_w_color_loss
            depth_mask = (target_d > 0)
            depth_loss = torch.abs(target_d[depth_mask] -
                                   depth[depth_mask]).sum()
            loss_dict['depth_loss'] = depth_loss
            if stage == 'color':
                color_loss = torch.abs(target_rgb - color).sum()
                weighted_color_loss = w_color_loss * color_loss
                loss_dict['rgb_loss'] = weighted_color_loss
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        decoders_para_list = []
        if not self.config.mapping_fix_fine:
            decoders_para_list += list(self.decoder.fine_decoder.parameters())
        if not self.config.mapping_fix_color:
            decoders_para_list += list(self.decoder.color_decoder.parameters())
        param_groups['decoder'] = decoders_para_list
        # grid_params
        for key, grid in self.grid_c.items():
            grid = grid.to(self.device)
            if (self.config.mapping_frustum_feature_selection
                    and not self.config.coarse):
                mask = self.grid_opti_mask[key]
                grid.set_mask(mask)
            param_groups[key] = list(grid.parameters())
        return param_groups

    # only used by mesher
    def query_fn(self, pi):
        """Evaluates the occupancy value for the points.

        Args:
            p (tensor, N*3): point coordinates.

        Returns:
            ret (tensor): occupancy value of input points.
        """
        pi.unsqueeze(0)
        ret = self.decoder(pi, c_grid=self.grid_c, stage='fine')
        ret = ret.squeeze(0)
        if len(ret.shape) == 1 and ret.shape[0] == 4:
            ret = ret.unsqueeze(0)
        return ret

    # only used by mesher
    def color_func(self, pi):
        pi.unsqueeze(0)
        ret = self.decoder(pi, c_grid=self.grid_c, stage='color')
        ret = ret.squeeze(0)
        if len(ret.shape) == 1 and ret.shape[0] == 4:
            ret = ret.unsqueeze(0)
        return ret

    def get_decoder(self):
        """Get the decoder of the scene representation."""
        self.decoder = NICE(
            dim=self.config.data_dim,
            c_dim=self.config.model_c_dim,
            coarse=self.config.coarse,
            coarse_grid_len=self.config.grid_len_coarse,
            middle_grid_len=self.config.grid_len_middle,
            fine_grid_len=self.config.grid_len_fine,
            color_grid_len=self.config.grid_len_color,
            pos_embedding_method=self.config.model_pos_embedding_method)

    def grid_init(self):
        """Initialize the hierarchical feature grids."""
        if self.config.coarse:
            coarse_grid_len = self.config.grid_len_coarse

        middle_grid_len = self.config.grid_len_middle
        fine_grid_len = self.config.grid_len_fine
        color_grid_len = self.config.grid_len_color

        self.grid_c = {}
        c_dim = self.config.model_c_dim
        xyz_len = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        if self.config.coarse:
            coarse_key = 'grid_coarse'
            self.grid_c[coarse_key] = FeatureGrid(
                xyz_len=xyz_len * self.config.model_coarse_bound_enlarge,
                grid_len=coarse_grid_len,
                c_dim=c_dim,
                std=0.01)

        middle_key = 'grid_middle'
        self.grid_c[middle_key] = FeatureGrid(xyz_len=xyz_len,
                                              grid_len=middle_grid_len,
                                              c_dim=c_dim,
                                              std=0.01)

        fine_key = 'grid_fine'
        self.grid_c[fine_key] = FeatureGrid(xyz_len=xyz_len,
                                            grid_len=fine_grid_len,
                                            c_dim=c_dim,
                                            std=0.0001)

        color_key = 'grid_color'
        self.grid_c[color_key] = FeatureGrid(xyz_len=xyz_len,
                                             grid_len=color_grid_len,
                                             c_dim=c_dim,
                                             std=0.01)

    def load_pretrain(self):
        """Load parameters of pretrained ConvOnet checkpoints to the
        decoders."""
        if self.config.coarse:
            ckpt = torch.load(self.config.pretrained_decoders_coarse,
                              map_location=self.device)
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.decoder.coarse_decoder.load_state_dict(coarse_dict)

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
        self.decoder.middle_decoder.load_state_dict(middle_dict)
        self.decoder.fine_decoder.load_state_dict(fine_dict)

    def load_bound(self):
        bound_divisible = self.config.grid_bound_divisible
        # enlarge the bound a bit to allow it divisible by bound_divisible
        self.bounding_box[:, 1] = (
            ((self.bounding_box[:, 1] - self.bounding_box[:, 0]) /
             bound_divisible).int() +
            1) * bound_divisible + self.bounding_box[:, 0]
        self.decoder.bound = self.bounding_box
        self.decoder.middle_decoder.bound = self.bounding_box
        self.decoder.fine_decoder.bound = self.bounding_box
        self.decoder.color_decoder.bound = self.bounding_box
        if self.config.coarse:
            self.decoder.coarse_decoder.bound = self.bounding_box * \
                self.config.model_coarse_bound_enlarge

    def eval_points(self, p, stage='color'):
        """Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            stage (str, optional): Query stage, corresponds to different
                levels. Defaults to 'color'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.config.points_batch_size)
        bound = self.bounding_box
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            ret = self.decoder(pi, c_grid=self.grid_c, stage=stage)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)

        return ret

    def render_batch_ray(self, rays_o, rays_d, stage, gt_depth=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        '''
        N_samples = self.config.rendering_n_samples
        N_surface = self.config.rendering_n_surface
        N_importance = self.config.rendering_n_importance

        N_rays = rays_o.shape[0]

        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples * 0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bounding_box.unsqueeze(0).to(self.device) -
                 det_rays_o) / det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0, torch.max(gt_depth * 1.2))
        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(0., 1., steps=N_surface).to(
                    self.device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth
                # sensor readings, meaning colorize on interpolated geometry
                # region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample
                # near the surface, since it is not a good idea to sample 16
                # points near (half even behind) camera, for pixels with zero
                # depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(self.device)
                # empirical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(
                    self.device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[
                    gt_none_zero_mask, :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(0).repeat(
                    (~gt_none_zero_mask).sum(), 1)
                z_vals_surface[
                    ~gt_none_zero_mask, :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=self.device)

        if not self.config.rendering_lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        if self.config.rendering_perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(self.device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]

        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, stage)

        raw = raw.reshape(N_rays, N_samples + N_surface, -1)

        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw,
            z_vals,
            rays_d,
            occupancy=self.config.occupancy,
            device=self.device)
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid,
                                   weights[..., 1:-1],
                                   N_importance,
                                   det=(self.config.rendering_perturb == 0.),
                                   device=self.device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, stage)
            raw = raw.reshape(N_rays, N_samples + N_importance + N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw,
                z_vals,
                rays_d,
                occupancy=self.config.occupancy,
                device=self.device)

        # Return rendering outputs
        ret = {'rgb': color, 'depth': depth, 'uncertainty': uncertainty}

        return ret
