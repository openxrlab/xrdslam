import functools
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

from deepslam.common.camera import Camera
from deepslam.common.common import get_rays, get_samples
from deepslam.methods.base_method import Method, MethodConfig
from deepslam.model_components.voxel_helpers_voxfusion import (eval_points,
                                                               get_scores)


@dataclass
class VoxFusionConfig(MethodConfig):
    """VoxFusion  Config."""
    _target: Type = field(default_factory=lambda: VoxFusion)

    # sample
    mapping_sample: int = 2048
    min_sample_pixels: int = 100
    tracking_sample: int = 1024
    # render image
    ray_batch_size: int = 3000


class VoxFusion(Method):
    def __init__(self, config: VoxFusionConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.model = self.config.model.setup(camera=camera, bounding_box=None)
        self.model.to(device)
        self.precompute()
        self.bundle_adjust = True

    def precompute(self):
        K = np.eye(3)
        K[0, 0] = self.camera.fx
        K[1, 1] = self.camera.fy
        K[0, 2] = self.camera.cx
        K[1, 2] = self.camera.cy
        ix, iy = torch.meshgrid(torch.arange(self.camera.width),
                                torch.arange(self.camera.height),
                                indexing='xy')
        rays_d = torch.stack([(ix - K[0, 2]) / K[0, 0],
                              -(iy - K[1, 2]) / K[1, 1], -torch.ones_like(ix)],
                             -1).float()  # [H, W, 3]
        self.rays_d = rays_d
        # update pose_offset
        # self.pose_offset = self.model.pose_offset

    def get_model_input(self, optimize_frames, is_mapping):
        batch_rays_d_list = []
        batch_rays_o_list = []
        batch_gt_depth_list = []
        batch_gt_color_list = []

        pixs_per_image = self.config.tracking_sample
        if is_mapping:
            pixs_per_image = self.config.mapping_sample

        for frame in optimize_frames:
            gt_depth = frame.depth
            gt_color = frame.rgb
            c2w = frame.get_pose()
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                get_samples(self.camera,
                            pixs_per_image,
                            c2w,
                            gt_depth,
                            gt_color,
                            device=self.model.device)

            batch_rays_o_list.append(batch_rays_o.float())
            batch_rays_d_list.append(batch_rays_d.float())
            batch_gt_depth_list.append(batch_gt_depth.float())
            batch_gt_color_list.append(batch_gt_color.float())

        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)

        ret = {
            'rays_o': batch_rays_o,  # [N, 3]
            'rays_d': batch_rays_d,  # [N, 3]
            'target_s': batch_gt_color,  # [N, 3]
            'target_d': batch_gt_depth,  # [N, 1]
        }

        return ret

    def create_voxels(self, frame):
        '''
        Note: insert_points will update_map_states, and set map.leaf_num
        '''
        depth = torch.from_numpy(frame.depth)  # [H, W], on cpu
        points = self.rays_d * depth[..., None]
        points = points[depth > 0].reshape(-1, 3).cuda()
        pose = frame.get_pose().cuda()
        points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3,
                                                                3]  # [N, 3]
        self.model.insert_points(points)

    def pre_precessing(self, cur_frame, is_mapping):
        if is_mapping:
            self.create_voxels(cur_frame)

    def get_loss(self,
                 optimize_frames,
                 is_mapping,
                 step=None,
                 n_iters=None,
                 coarse=False):
        model_input = self.get_model_input(optimize_frames, is_mapping)
        model_outputs = self.model(model_input)
        loss_dict = self.model.get_loss_dict(model_outputs, model_input,
                                             is_mapping, step)
        loss = functools.reduce(torch.add, loss_dict.values())
        return loss

    def render_img(self, c2w, gt_depth=None, idx=None):
        with self.lock and torch.no_grad():
            rays_o, rays_d = get_rays(self.camera,
                                      c2w,
                                      device=self.model.device)

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            if gt_depth is not None:
                gt_depth = torch.from_numpy(gt_depth).to(
                    self.model.device).reshape(-1, 1)

            ray_batch_size = self.config.ray_batch_size
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]

                model_input_batch = {
                    'rays_o': rays_o_batch,
                    'rays_d': rays_d_batch,
                    'target_s': None,
                    'target_d': None,
                }

                if gt_depth is not None:
                    model_input_batch['target_d'] = gt_depth[i:i +
                                                             ray_batch_size]

                model_outputs = self.model(model_input_batch)

                depth_list.append(model_outputs['depth'].double())
                color_list.append(model_outputs['rgb'])

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(self.camera.height, self.camera.width)
            color = color.reshape(self.camera.height, self.camera.width, 3)

            return color.clone().squeeze().cpu().numpy(), depth.clone(
            ).squeeze().cpu().numpy()

    # provided by octree
    # TODO: use Mesher.get_mesh
    def get_mesh(self):
        with self.lock and torch.no_grad():
            return self.extract_mesh(clean_mesh=False,
                                     require_color=True,
                                     res=8)

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False, require_color=False):
        # get map states
        voxels, _, features, leaf_num = self.model.octree.get_all()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        centres = (voxels[:, :3] +
                   voxels[:, -1:] / 2) * self.model.config.voxel_size

        map_states = {}
        map_states['voxel_vertex_idx'] = features.cuda()
        map_states['voxel_center_xyz'] = centres.cuda()
        map_states['voxel_vertex_emb'] = self.model.embeddings.detach().clone(
        )  # .cuda()

        sdf_grid = get_scores(self.model.decoder.cuda(),
                              map_states,
                              self.model.config.voxel_size,
                              bits=res)
        sdf_grid = sdf_grid.reshape(-1, res, res, res, 4)

        voxel_centres = map_states['voxel_center_xyz']
        verts, faces = self.marching_cubes(
            voxel_centres,
            sdf_grid)  # [N_verts, 3], [N_face, 3 (3 ids of verts)]
        '''
        # TODO: clean mesh need depth and pose, use key frame info
        '''
        colors = None
        if require_color:
            verts_torch = torch.from_numpy(verts).float().cuda()
            batch_points = torch.split(verts_torch, 1000)
            colors = []
            for points in batch_points:  # points.shape: [N_points(1000), 3]
                voxel_pos = points // self.model.config.voxel_size
                batch_voxels = voxels[:, :3].cuda(
                )  # batch_voxels.shape: [N_voxels, 3]
                batch_voxels = batch_voxels.unsqueeze(0).repeat(
                    voxel_pos.shape[0], 1, 1)
                # filter outliers
                # 1. find each point in which voxel? (voxel id should be same.)
                nonzeros = (batch_voxels == voxel_pos.unsqueeze(1)).all(-1)
                nonzeros = torch.where(nonzeros,
                                       torch.ones_like(nonzeros).int(),
                                       -torch.ones_like(nonzeros).int()
                                       )  # shape: [N_points, N_voxels]
                # 2. sorting in descending: 1 is before -1
                sorted, index = torch.sort(nonzeros, dim=-1, descending=True)
                # 3. get the first voxel and judge validation
                # (-1 is not in the voxels)
                sorted = sorted[:, 0]
                index = index[:, 0]
                valid = (sorted != -1)
                color_empty = torch.zeros_like(points)
                points = points[valid, :]
                index = index[valid]
                # get color
                if len(points) > 0:
                    color = eval_points(self.model.decoder, map_states, points,
                                        index,
                                        self.model.config.voxel_size).cuda()
                    color_empty[valid] = color
                colors += [color_empty]
            colors = torch.cat(colors, 0)
        if require_color:
            mesh = trimesh.Trimesh(verts,
                                   faces,
                                   vertex_colors=colors.detach().cpu().numpy())
        else:
            mesh = trimesh.Trimesh(verts, faces, vertex_colors=None)
        # mesh.fix_normals()
        return mesh

    @torch.no_grad()
    def marching_cubes(self, voxels, sdf):
        voxels = voxels[:, :3]
        sdf = sdf[..., 3]
        res = 1.0 / (sdf.shape[1] - 1)
        spacing = [res, res, res]

        num_verts = 0
        total_verts = []
        total_faces = []
        for i in range(len(voxels)):
            sdf_volume = sdf[i].detach().cpu().numpy()
            if np.min(sdf_volume) > 0 or np.max(sdf_volume) < 0:
                continue
            verts, faces, _, _ = marching_cubes(sdf_volume, 0, spacing=spacing)
            verts -= 0.5
            verts *= self.model.config.voxel_size
            verts += voxels[i].detach().cpu().numpy()
            faces += num_verts
            num_verts += verts.shape[0]

            total_verts += [verts]
            total_faces += [faces]
        total_verts = np.concatenate(total_verts)
        total_faces = np.concatenate(total_faces)
        return total_verts, total_faces
