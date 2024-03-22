from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.model_components.decoder_coslam import ColorSDFNet, ColorSDFNet_v2
from slam.model_components.encodings_coslam import get_encoder
from slam.model_components.utils import (batchify, compute_loss, coordinates,
                                         get_sdf_loss, sample_pdf)
from slam.models.base_model import Model, ModelConfig


@dataclass
class JointEncodingConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: JointEncoding)
    # grid
    voxel_sdf: float = 0.02
    voxel_color: float = 0.08
    enc: str = 'HashGrid'
    pos_enc: str = 'OneBlob'
    pos_nbins: int = 16
    hashsize: int = 16
    oneGrid: bool = True
    # decoder
    geo_feat_dim: int = 15
    hidden_dim: int = 32
    num_layers: int = 2
    num_layers_color: int = 2
    hidden_dim_color: int = 32
    tcnn_network: bool = False
    tcnn_encoding: bool = False

    # train
    trainging_rgb_weight: float = 5.0
    trainging_depth_weight: float = 0.1
    trainging_sdf_weight: float = 1000
    trainging_fs_weight: float = 10
    trainging_smooth_weight: float = 0.000001
    trainging_smooth_pts: int = 32
    trainging_smooth_vox: float = 0.1
    trainging_smooth_margin: float = 0.05
    training_n_samples: int = 256
    training_n_sample_d: int = 32
    training_range_d: float = 0.1
    training_n_range_d: int = 11
    training_n_importance: int = 0
    training_perturb: int = 1
    training_white_bkgd: bool = False
    training_trunc: float = 0.1
    training_rgb_missing: float = 0.05

    # data
    data_sc_factor: int = 1
    data_translation: int = 0

    # cam
    cam_near: float = 0.0
    cam_far: float = 5.0
    cam_depth_trunc: float = 100.0

    # mesh
    mesh_render_color: bool = False


class JointEncoding(Model):
    """Model class."""

    config: JointEncodingConfig

    def __init__(
        self,
        config: JointEncodingConfig,
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
        self.get_resolution()
        self.get_encoding()
        self.get_decoder()

    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        target_d = inputs['target_d']
        target_rgb = inputs['target_s']
        first_flag = inputs['first']

        valid_depth_mask = (target_d.squeeze() > 0.) * (
            target_d.squeeze() < self.config.cam_depth_trunc)
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight == 0] = self.config.training_rgb_missing

        # Get render loss
        rgb_loss = compute_loss(outputs['rgb'] * rgb_weight,
                                target_rgb * rgb_weight)
        depth_loss = compute_loss(outputs['depth'].squeeze()[valid_depth_mask],
                                  target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in outputs:
            rgb_loss += compute_loss(outputs['rgb0'] * rgb_weight,
                                     target_rgb * rgb_weight)
            depth_loss += compute_loss(outputs['depth0'][valid_depth_mask],
                                       target_d.squeeze()[valid_depth_mask])

        # Get sdf loss
        z_vals = outputs['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = outputs['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.config.training_trunc * self.config.data_sc_factor
        fs_loss, sdf_loss = get_sdf_loss(z_vals,
                                         target_d,
                                         sdf,
                                         truncation,
                                         'l2',
                                         grad=None)

        loss_dict = {
            'rgb_loss': rgb_loss * self.config.trainging_rgb_weight,
            'depth_loss': depth_loss * self.config.trainging_depth_weight,
            'sdf_loss': sdf_loss * self.config.trainging_sdf_weight,
            'fs_loss': fs_loss * self.config.trainging_fs_weight,
        }

        if is_mapping and not first_flag:
            smooth_loss = self.smoothness(self.config.trainging_smooth_pts,
                                          self.config.trainging_smooth_vox,
                                          self.config.trainging_smooth_margin)
            loss_dict['smooth_loss'] = smooth_loss * \
                self.config.trainging_smooth_weight

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['decoder'] = list(self.decoder.parameters())
        param_groups['embed_fn'] = list(self.embed_fn.parameters())
        if not self.config.oneGrid:
            param_groups['embed_fn_color'] = list(
                self.embed_fn_color.parameters())
        return param_groups

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        rays_o = input['rays_o']
        rays_d = input['rays_d']
        target_d = input['target_d']
        outputs = self.render_rays(rays_o, rays_d, target_d=target_d)
        return outputs

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Smoothness loss of feature grid.
        """
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bounding_box[:, 1] - \
            self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu',
                             flatten=False).float().to(volume)
        pts = (coords + torch.rand(
            (1, 1, 1,
             3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        pts_tcnn = pts
        if self.config.tcnn_encoding:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (
                self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pts_tcnn = pts_tcnn.to(self.device)

        sdf = self.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        loss = (tv_x + tv_y + tv_z) / (sample_points**3)

        return loss

    def get_resolution(self):
        """Get the resolution of the grid."""
        dim_max = (self.bounding_box[:, 1] - self.bounding_box[:, 0]).max()
        if self.config.voxel_sdf > 10:
            self.resolution_sdf = self.config.voxel_sdf
        else:
            self.resolution_sdf = int(dim_max / self.config.voxel_sdf)

        if self.config.voxel_color > 10:
            self.resolution_color = self.config.voxel_color
        else:
            self.resolution_color = int(dim_max / self.config.voxel_color)

    def get_encoding(self):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Get the encoding of the scene representation.
        """
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(
            self.config.pos_enc, n_bins=self.config.pos_nbins)

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(
            self.config.enc,
            log2_hashmap_size=self.config.hashsize,
            desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.config.oneGrid:
            print('Color resolution:', self.resolution_color)
            self.embed_fn_color, self.input_ch_color = get_encoder(
                self.config.enc,
                log2_hashmap_size=self.config.hashsize,
                desired_resolution=self.resolution_color)

    def get_decoder(self):
        """Get the decoder of the scene representation."""
        if not self.config.oneGrid:
            self.decoder = ColorSDFNet(self.config,
                                       input_ch=self.input_ch,
                                       input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(self.config,
                                          input_ch=self.input_ch,
                                          input_ch_pos=self.input_ch_pos)

        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    def render_rays(self, rays_o, rays_d, target_d=None):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]
        """

        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(
                -self.config.training_range_d,
                self.config.training_range_d,
                steps=self.config.training_n_range_d).to(target_d)
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze() <= 0] = torch.linspace(
                self.config.cam_near,
                self.config.cam_far,
                steps=self.config.training_n_range_d).to(target_d)

            if self.config.training_n_sample_d > 0:
                z_vals = torch.linspace(
                    self.config.cam_near, self.config.cam_far,
                    self.config.training_n_sample_d)[None, :].repeat(
                        n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config.cam_near, self.config.cam_far,
                                    self.config.training_n_samples).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1)  # [n_rays, n_samples]

        # Perturb sampling depths
        if self.config.training_perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(
                z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = \
            self.raw2outputs(raw, z_vals, self.config.training_white_bkgd)

        # Importance sampling
        if self.config.training_n_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = \
                rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid,
                                   weights[..., 1:-1],
                                   self.config.training_n_importance,
                                   det=(self.config.training_perturb == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
                ..., :, None]  # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts)

            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = \
                self.raw2outputs(raw, z_vals, self.config.training_white_bkgd)

        # Return rendering outputs
        ret = {
            'rgb': rgb_map,
            'depth': depth_map,
            'disp_map': disp_map,
            'acc_map': acc_map,
            'depth_var': depth_var,
        }
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.config.training_n_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret

    def sdf2weights(self, sdf, z_vals):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        """
        weights = torch.sigmoid(
            sdf / self.config.training_trunc) * torch.sigmoid(
                -sdf / self.config.training_trunc)

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs),
                           torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)  # The first surface
        mask = torch.where(
            z_vals <
            z_min + self.config.data_sc_factor * self.config.training_trunc,
            torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        """
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights *
                              torch.square(z_vals - depth_map.unsqueeze(-1)),
                              dim=-1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                  depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var

    # only used by mesher
    def query_fn(self, pi):
        if self.config.tcnn_encoding:
            bounding_box = self.bounding_box.to(self.device)
            pi = (pi - bounding_box[:, 0]) / (bounding_box[:, 1] -
                                              bounding_box[:, 0])
        pi = pi.unsqueeze(1)
        return self.query_sdf(pi)

    # only used by mesher
    def color_func(self, pi):
        if self.config.tcnn_encoding:
            bounding_box = self.bounding_box.to(self.device)
            pi = (pi - bounding_box[:, 0]) / (bounding_box[:, 1] -
                                              bounding_box[:, 0])
        pi = pi.unsqueeze(1)
        return self.query_color(pi)

    def query_sdf(self, query_points, return_geo=False, embed=False):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        """
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(
                embedded,
                list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(
            geo_feat,
            list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat

    def query_color(self, query_points):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0."""
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])

    def query_color_sdf(self, query_points):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        if not self.config.oneGrid:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)

    def run_network(self, inputs):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        # Normalize the input to [0, 1] (TCNN convention)
        if self.config.tcnn_encoding:
            self.bounding_box = self.bounding_box.to(self.device)
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (
                self.bounding_box[:, 1] - self.bounding_box[:, 0])

        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        outputs = torch.reshape(
            outputs_flat,
            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs

    def render_surface_color(self, rays_o, normal):
        """This function is modified from co-slam, licensed under the Apache
        License, Version 2.0.

        Render the surface color of the points.

        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        """
        n_rays = rays_o.shape[0]
        trunc = self.config.training_trunc
        z_vals = torch.linspace(
            -trunc, trunc, steps=self.config.training_n_range_d).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline

        pts = rays_o[..., :] + normal[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb, disp_map, acc_map, weights, depth_map, depth_var = \
            self.raw2outputs(raw, z_vals, self.config.training_white_bkgd)
        return rgb
