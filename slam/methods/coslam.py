import functools
import random
from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
import torch

from slam.common.camera import Camera
from slam.common.common import get_rays, get_samples
from slam.common.mesher import MesherConfig
from slam.engine.optimizers import Optimizers
from slam.methods.base_method import Method, MethodConfig
from slam.utils.utils import get_camera_rays


@dataclass
class CoSLAMConfig(MethodConfig):
    """CoSLAM  Config."""
    _target: Type = field(default_factory=lambda: CoSLAM)
    # mesher
    mesher: MesherConfig = MesherConfig()

    rays_to_save_ratio: float = 0.05
    # tracking
    tracking_Wedge: int = 20
    tracking_Hedge: int = 20
    # sample
    mapping_sample: int = 2048
    min_sample_pixels: int = 100
    tracking_sample: int = 1024
    # render image
    ray_batch_size: int = 3000
    # bound
    marching_cubes_bound: List[List[float]] = field(
        default_factory=lambda: [[-3.5, 3], [-3, 3], [-3, 3]])
    mapping_bound: List[List[float]] = field(
        default_factory=lambda: [[-3.5, 3], [-3, 3], [-3, 3]])


class CoSLAM(Method):
    def __init__(self, config: CoSLAMConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.marching_cube_bound = torch.from_numpy(
            np.array(self.config.marching_cubes_bound))
        self.bounding_box = torch.from_numpy(
            np.array(self.config.mapping_bound))
        self.model = self.config.model.setup(camera=camera,
                                             bounding_box=self.bounding_box)
        self.model.to(device)
        self.mesher = self.config.mesher.setup(
            camera=camera,
            bounding_box=self.bounding_box,
            marching_cubes_bound=self.marching_cube_bound)

        self.cur_mesh = None
        self.bundle_adjust = True

        self.num_rays_to_save = int(self.camera.width * self.camera.height *
                                    self.config.rays_to_save_ratio)
        self.rays = None

        self.model_optimizers = None

    def setup_optimizers(self,
                         n_iters,
                         optimize_frames,
                         is_mapping=True,
                         coarse=False) -> Optimizers:

        # update optimizer config by n_iters
        optimizer_config = self.config.optimizers.copy()
        if not is_mapping:
            if self.config.separate_LR:
                pose_params = {'tracking_pose_r': [], 'tracking_pose_t': []}
                for keyframe in optimize_frames:
                    pose_params['tracking_pose_r'].extend(
                        [keyframe.get_params()[0]])
                    pose_params['tracking_pose_t'].extend(
                        [keyframe.get_params()[1]])
                    return Optimizers(optimizer_config, {**pose_params})
            else:
                pose_params = {'tracking_pose': []}
                for keyframe in optimize_frames:
                    pose_params['tracking_pose'].extend(keyframe.get_params())
                    return Optimizers(optimizer_config, {**pose_params})
        else:
            if not self.bundle_adjust or len(optimize_frames) == 1:
                if self.model_optimizers is None:
                    model_params = self.model.get_param_groups()
                    self.model_optimizers = Optimizers(optimizer_config,
                                                       {**model_params})
                return self.model_optimizers
            else:
                if self.config.separate_LR:
                    pose_params = {'mapping_pose_r': [], 'mapping_pose_t': []}
                else:
                    pose_params = {'mapping_pose': []}
                # fix the first frame's pose
                for keyframe in optimize_frames[1:]:
                    if self.config.separate_LR:
                        pose_params['mapping_pose_r'].extend(
                            [keyframe.get_params()[0]])
                        pose_params['mapping_pose_t'].extend(
                            [keyframe.get_params()[1]])
                    else:
                        pose_params['mapping_pose'].extend(
                            keyframe.get_params())
                pose_optimizers = Optimizers(optimizer_config, {**pose_params})
                optimizers = pose_optimizers + self.model_optimizers
                return optimizers

    def sample_single_keyframe_rays(self, keyframe, bs):
        gt_depth = torch.from_numpy(keyframe.depth.astype(np.float32))
        gt_color = torch.from_numpy(keyframe.rgb.astype(np.float32))
        all_rays_d = get_camera_rays(self.camera.height, self.camera.width,
                                     self.camera.fx, self.camera.fy,
                                     self.camera.cx, self.camera.cy)
        rays = torch.cat([all_rays_d, gt_color, gt_depth[..., None]], dim=-1)
        rays = rays.reshape(-1, rays.shape[-1])
        idxs = random.sample(range(0, self.camera.height * self.camera.width),
                             bs)
        rays = rays[idxs]
        return rays  # [bs, 7]

    def add_keyframe(self, keyframe):
        with self.lock:
            rays = self.sample_single_keyframe_rays(
                keyframe, self.num_rays_to_save)  # [N, 7]
            # only save pose and rays
            if self.rays is None:
                self.rays = torch.cat([rays])
            else:
                rays_list = [self.rays, rays]
                self.rays = torch.cat(rays_list, dim=0)
                # self.rays = torch.cat([self.rays, rays], dim=0)
            # no need to save rgb and depth
            keyframe.rgb = None
            keyframe.depth = None
            self.keyframe_graph.append(keyframe)
            torch.cuda.empty_cache()

    def sample_global_rays(self, bs):
        num_kf = len(self.keyframe_graph)
        idxs = torch.tensor(
            random.sample(range(num_kf * self.num_rays_to_save), bs))
        sample_rays = self.rays.reshape(-1, 7)[idxs]
        select_frame_ids = idxs // self.num_rays_to_save
        return sample_rays, select_frame_ids

    def get_model_input(self, optimize_frames, is_mapping):
        cur_frame = optimize_frames[-1]
        if is_mapping:
            ids_all = []
            rays_all = []
            poses_all = []
            pixs_cur_image = self.config.mapping_sample
            if len(self.keyframe_graph) > 0:
                # keyframes' rays
                sample_rays, select_frame_ids = self.sample_global_rays(
                    self.config.mapping_sample)
                ids_all = [select_frame_ids]
                rays_all = [sample_rays]
                for frame in optimize_frames[:-1]:
                    pose = frame.get_pose().unsqueeze(0).to(self.device)
                    # fix the first frame's pose
                    if frame.fid == 0:
                        pose = pose.detach()
                    poses_all.append(pose)
                pixs_cur_image = np.maximum(
                    self.config.mapping_sample // len(self.keyframe_graph),
                    self.config.min_sample_pixels)

            # curframe's rays
            rays = self.sample_single_keyframe_rays(cur_frame,
                                                    pixs_cur_image)  # [N, 7]

            # keyframes + curframe
            poses_all.append(cur_frame.get_pose().unsqueeze(0).to(self.device))
            ids_all.append(-torch.ones((len(rays))).to(torch.int64))
            rays_all.append(rays)

            poses_all = torch.cat(poses_all, dim=0)
            ids_all = torch.cat(ids_all, dim=0)
            rays_all = torch.cat(rays_all, dim=0)

            rays_d_cam = rays_all[..., :3].to(self.device)  # [N, 3]
            target_s = rays_all[..., 3:6].to(self.device)  # [N, 3]
            target_d = rays_all[..., 6:7].to(self.device).reshape(-1,
                                                                  1)  # [N, 1]

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(
                rays_d_cam[..., None, None, :] *
                poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3,
                               -1].repeat(1, rays_d.shape[1],
                                          1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)  # [N, 3]

            batch_rays_d = rays_d
            batch_rays_o = rays_o
            batch_gt_depth = target_d
            batch_gt_color = target_s
            first_flag = True
            if len(self.keyframe_graph) > 0:
                first_flag = False
        else:
            gt_depth = cur_frame.depth
            gt_color = cur_frame.rgb
            c2w = cur_frame.get_pose()
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                get_samples(self.camera,
                            self.config.tracking_sample,
                            c2w,
                            gt_depth,
                            gt_color,
                            device=self.model.device,
                            Hedge=self.config.tracking_Hedge,
                            Wedge=self.config.tracking_Wedge)
            first_flag = False

        return {
            'rays_o': batch_rays_o.float(),
            'rays_d': batch_rays_d.float(),
            'target_s': batch_gt_color.float(),
            'target_d': batch_gt_depth.float(),
            'first': first_flag,
        }

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

    def get_mesh(self):
        with self.lock:
            self.cur_mesh = self.mesher.get_mesh(
                keyframe_graph=self.keyframe_graph,
                query_fn=self.model.query_fn,
                color_func=self.model.color_func,
                device=self.device)
            return self.cur_mesh
