from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.model_components.decoder_voxfusion import Decoder
from slam.model_components.utils import compute_loss, get_sdf_loss
from slam.model_components.voxel_helpers_voxfusion import (get_features,
                                                           masked_scatter,
                                                           masked_scatter_ones,
                                                           ray, ray_intersect,
                                                           ray_sample)
from slam.models.base_model import Model, ModelConfig


# load .so
def find_so_files(directory):
    so_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.so'):
                so_files.append(os.path.join(root, file))
    return so_files


search_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../third_party/sparse_octforest/build/')
so_files = find_so_files(search_directory)
for so_file in so_files:
    torch.classes.load_library(so_file)


@dataclass
class SparseVoxelConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: SparseVoxel)

    # octree
    voxels_each_dim: int = 256  # number of voxels for each dim in octree
    voxel_size: float = 0.2  # in [meter]
    num_embeddings: int = 20000
    embed_dim: int = 16
    max_distance: int = 10
    max_dpeth: float = 10  # different in different scene

    # training weight
    training_trunc: float = 0.05
    trainging_rgb_weight: float = .5  # 1.0
    trainging_depth_weight: float = 1.0
    trainging_sdf_weight: float = 5000
    trainging_fs_weight: float = 10.0

    # decoder
    depth: int = 2
    width: int = 128
    in_dim: int = 16
    embedder: str = 'none'  #

    # tracking and mapping params
    step_size: float = 0.05  # replica:0.1 scannet:0.05
    max_voxel_hit: int = 20
    num_iterations: int = 30
    overlap_th: float = 0.7
    keyframe_th: int = 30
    keyframe_selection: str = 'random'

    # data
    data_sc_factor: int = 1


class SparseVoxel(Model):
    """Model class."""

    config: SparseVoxelConfig

    def __init__(
        self,
        config: SparseVoxelConfig,
        camera: Camera,
        bounding_box,
        **kwargs,
    ) -> None:
        super().__init__(config=config,
                         camera=camera,
                         bounding_box=bounding_box,
                         **kwargs)
        self.map_lock = torch.multiprocessing.RLock()
        self.config.step_size = self.config.voxel_size * self.config.step_size
        self.pose_offset = int(self.config.voxels_each_dim / 2.0 *
                               self.config.voxel_size)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()
        self.get_octree()
        self.get_decoder()

    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        """color and depth are filled, use 'ray_mask' to select valid rays."""
        loss_dict = {}

        # select by svo intersection
        ray_mask = outputs['ray_mask']
        target_d = inputs['target_d'][ray_mask]
        target_rgb = inputs['target_s'][ray_mask]

        valid_depth_mask = (target_d.squeeze() > 0.01) * (
            target_d.squeeze() < self.config.max_dpeth)
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)

        # Get render loss
        rgb_loss = compute_loss(outputs['rgb'][ray_mask] * rgb_weight,
                                target_rgb * rgb_weight,
                                loss_type='l1')
        depth_loss = compute_loss(
            outputs['depth'][ray_mask].squeeze()[valid_depth_mask],
            target_d.squeeze()[valid_depth_mask],
            loss_type='l1')

        # Get sdf loss
        z_vals = outputs['z_vals']
        sdf = outputs['sdf']
        truncation = self.config.training_trunc * self.config.data_sc_factor
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation,
                                         'l2')

        loss_dict = {
            'rgb_loss': rgb_loss * self.config.trainging_rgb_weight,
            'depth_loss': depth_loss * self.config.trainging_depth_weight,
            'sdf_loss': sdf_loss * self.config.trainging_sdf_weight,
            'fs_loss': fs_loss * self.config.trainging_fs_weight,
        }

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['decoder'] = list(self.decoder.parameters())
        param_groups['embeddings'] = [self.embeddings]

        return param_groups

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        """model.forward() -> get_outputs."""
        rays_o = input['rays_o'].unsqueeze(0)  # [N, 3] -> [1,N,3]
        rays_d = input['rays_d'].unsqueeze(0)
        target_d = input['target_d'].unsqueeze(0)
        outputs = self.render_rays(rays_o, rays_d, target_d=target_d)
        return outputs

    def render_rays(self, rays_o, rays_d, target_d=None, chunk_size=-1):
        """This function is modified from voxfusion,

        intersections -> sample feature -> decoder
        rays_o: # [1, N, 3], need , dim 0 = 1, for voxel operation
        rays_d: # [1, N, 3]
        """
        map_states = self.map_states
        centres = map_states['voxel_center_xyz']  # [1, N_1, 3]
        children = map_states['voxel_structure']

        # perform ray & voxel intersect
        intersections, hits = ray_intersect(rays_o, rays_d, centres, children,
                                            self.config.voxel_size,
                                            self.config.max_voxel_hit,
                                            self.config.max_distance)

        # NOTE: if no hit, pose translation is wrong.
        # assert(hits.sum() > 0)
        if hits.sum() == 0:
            print('\n\n', '!' * 20, 'render_rays. no hit', '!' * 20, '\n\n')
            return None

        # sample points along ray
        ray_mask = hits.view(1, -1)
        intersections = {
            name: outs[ray_mask].reshape(-1, outs.size(-1))
            for name, outs in intersections.items()
        }

        rays_o = rays_o[ray_mask].reshape(-1, 3)
        rays_d = rays_d[ray_mask].reshape(-1, 3)

        samples = ray_sample(intersections, step_size=self.config.step_size)

        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()

        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if sample_mask.sum() == 0:  # miss everything skip
            return None

        sampled_xyz = ray(rays_o.unsqueeze(1), rays_d.unsqueeze(1),
                          sampled_depth.unsqueeze(2))
        sampled_dir = rays_d.unsqueeze(1).expand(*sampled_depth.size(),
                                                 rays_d.size()[-1])
        sampled_dir = sampled_dir / (
            torch.norm(sampled_dir, 2, -1, keepdim=True) + 1e-8)
        samples['sampled_point_xyz'] = sampled_xyz
        samples['sampled_point_ray_direction'] = sampled_dir

        # apply mask
        samples_valid = {name: s[sample_mask] for name, s in samples.items()}

        num_points = samples_valid['sampled_point_depth'].shape[0]
        field_outputs = []
        if chunk_size < 0:
            chunk_size = num_points

        for i in range(0, num_points, chunk_size):
            chunk_samples = {
                name: s[i:i + chunk_size]
                for name, s in samples_valid.items()
            }

            # get encoder features as inputs
            chunk_inputs = get_features(chunk_samples, map_states,
                                        self.config.voxel_size)

            # forward implicit fields
            chunk_outputs = self.decoder(chunk_inputs)

            field_outputs.append(chunk_outputs)

        field_outputs = {
            name: torch.cat([r[name] for r in field_outputs], dim=0)
            for name in field_outputs[0]
        }

        outputs = {'sample_mask': sample_mask}

        sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(
            -1)  # [masked_N_rays] -> [masked_N_rays, 1]
        colour = masked_scatter(sample_mask,
                                field_outputs['color'])  # [masked_N_rays, 3]
        sample_mask = outputs['sample_mask']  # [masked_N_rays, N_samples]

        valid_mask = torch.where(sample_mask, torch.ones_like(sample_mask),
                                 torch.zeros_like(sample_mask))

        z_vals = samples['sampled_point_depth']

        weights, z_min = self.sdf2weights(sdf, z_vals, valid_mask)
        rgb = torch.sum(weights[..., None] * colour, dim=-2)
        depth = torch.sum(weights * z_vals, dim=-1)

        # fill_in: masked_N_rays -> N_rays
        ray_mask = ray_mask.squeeze()  # [1, N_rays] -> [N_rays]
        depth = (depth.new_ones(ray_mask.shape[0]) * 0.0).masked_scatter(
            ray_mask, depth)
        rgb = (rgb.new_ones((ray_mask.shape[0], 3)) * 0.0).masked_scatter(
            ray_mask.unsqueeze(-1).expand((ray_mask.shape[0], 3)), rgb)

        ret = {
            'depth': depth,  # [masked_N_rays]
            'rgb': rgb,  # [masked_N_rays, 3]
            'sdf': sdf,  # [masked_N_rays, N_samples]
            'z_vals': z_vals,  # [masked_N_rays, N_samples]
            'ray_mask': ray_mask,  # [N_rays]
            'weights': weights,  # [masked_N_rays, N_samples]
            'z_min': z_min,  # [masked_N_rays, 1]
        }

        return ret

    def sdf2weights(self, sdf, z_vals, valid_mask):
        """This function is modified from voxfusion,

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

        weights = weights * mask * valid_mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) +
                          1e-8), z_min

    def get_octree(self):
        self.octree = torch.classes.forest.Octree(self.config.voxels_each_dim)
        self.embeddings = torch.nn.Parameter(torch.zeros(
            (self.config.num_embeddings, self.config.embed_dim),
            dtype=torch.float32,
            device='cuda',
        ),
                                             requires_grad=True)
        torch.nn.init.normal_(self.embeddings, std=0.01)
        self.embeddings.requires_grad = True

    def get_decoder(self):
        """Get the decoder of the scene representation."""
        self.decoder = Decoder(depth=self.config.depth,
                               width=self.config.width,
                               in_dim=self.config.embed_dim,
                               embedder=self.config.embedder)

    def insert_points(self, points):
        """Compute corresponding voxels from points, and insert them in svo,
        set largest voxel_num to avoid."""
        voxels = torch.div(points,
                           self.config.voxel_size,
                           rounding_mode='floor')
        voxels = torch.unique(voxels.cpu().int(), sorted=False, dim=0)
        # here, voxels.cpu().int() and (voxels.cpu().int()[:, None]).view(-1,3)
        # has the same shape: [N_points, 3]
        # i think we can remove repeated voxel ids to reduce insert time and
        # torch loading time for svo.
        self.octree.insert(voxels)
        self.update_map_states()

    @torch.enable_grad()
    def update_map_states(self):
        """This function is modified from voxfusion."""
        voxels, children, features, leaf_num = self.octree.get_all()
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.config.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_state = {}
        map_state['voxel_vertex_idx'] = features.cuda()
        map_state['voxel_center_xyz'] = centres
        map_state['voxel_structure'] = children
        map_state['voxel_vertex_emb'] = self.embeddings

        with self.map_lock:
            self.map_states = map_state

    def get_map_states(self):
        with self.map_lock:
            map_state = self.map_states.copy()

        return map_state
