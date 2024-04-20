import functools
from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
import torch

from slam.algorithms.base_algorithm import Algorithm, AlgorithmConfig
from slam.common.camera import Camera
from slam.common.common import get_rays, get_samples
from slam.common.mesher import MesherConfig


@dataclass
class NiceSLAMConfig(AlgorithmConfig):
    """NiceSLAM  Config."""
    _target: Type = field(default_factory=lambda: NiceSLAM)

    # mesher
    mesher: MesherConfig = MesherConfig()

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

    # tracking
    tracking_Wedge: int = 100
    tracking_Hedge: int = 100

    # mapping
    mapping_middle_iter_ratio: float = 0.4
    mapping_fine_iter_ratio: float = 0.6

    mapping_lr_factor: float = 1.0
    mapping_lr_first_factor: float = 5.0

    mapping_color_refine: bool = True


class NiceSLAM(Algorithm):

    config: NiceSLAMConfig

    def __init__(self, config: NiceSLAMConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.stage = 'color'
        self.marching_cube_bound = torch.from_numpy(
            np.array(self.config.marching_cubes_bound))
        self.bounding_box = torch.from_numpy(
            np.array(self.config.mapping_bound))
        self.config.model.coarse = self.config.coarse
        self.model = self.config.model.setup(camera=camera,
                                             bounding_box=self.bounding_box)
        self.model.to(device)
        self.mesher = self.config.mesher.setup(
            camera=camera,
            bounding_box=self.bounding_box,
            marching_cubes_bound=self.marching_cube_bound)

        self.cur_mesh = None

    def do_mapping(self, cur_frame):
        if not self.is_initialized():
            mapping_n_iters = self.config.mapping_first_n_iters
        else:
            mapping_n_iters = self.config.mapping_n_iters

        # here provides a color refinement postprocess
        if cur_frame.is_final_frame and self.config.mapping_color_refine:
            outer_joint_iters = 5
            self.config.mapping_window_size *= 2
            self.config.mapping_middle_iter_ratio = 0.0
            self.config.mapping_fine_iter_ratio = 0.0
            self.model.config.mapping_fix_color = True
            self.model.config.mapping_frustum_feature_selection = False
        else:
            outer_joint_iters = 1

        for _ in range(outer_joint_iters):
            # select optimize frames
            with torch.no_grad():
                optimize_frames = self.select_optimize_frames(
                    cur_frame,
                    keyframe_selection_method=self.config.
                    keyframe_selection_method)
            # optimize keyframes_pose, model_params, update model params
            self.optimize_update(mapping_n_iters,
                                 optimize_frames,
                                 is_mapping=True,
                                 coarse=False)

        # do coarse_mapper
        if self.config.coarse:
            optimize_frames = self.select_optimize_frames(
                cur_frame, keyframe_selection_method='random')
            self.optimize_update(mapping_n_iters,
                                 optimize_frames,
                                 is_mapping=True,
                                 coarse=True)

        if not self.is_initialized():
            self.set_initialized()

    def optimizer_config_update(self, max_iters, coarse=False):
        if len(self.keyframe_graph) > 4 and not coarse:
            self.bundle_adjust = True
        else:
            self.bundle_adjust = False
        for param_group_name, params in self.config.optimizers.items():
            lr_factor = self.config.mapping_lr_factor if (
                self.is_initialized() or 'pose'
                in param_group_name) else self.config.mapping_lr_first_factor
            if params['scheduler'] is not None:
                params['optimizer'].lr = lr_factor
                params['scheduler'].max_steps = max_iters
                # make sure iter_ration are same as those in NiceSLAMConfig
                params['scheduler'].coarse = coarse
                params['scheduler'].middle_iter_ratio = \
                    self.config.mapping_middle_iter_ratio
                params['scheduler'].fine_iter_ratio = \
                    self.config.mapping_fine_iter_ratio

    def pre_precessing(self, cur_frame, is_mapping):
        if is_mapping:
            self.model.pre_precessing(cur_frame)

    def post_processing(self, step, is_mapping, optimizer=None, coarse=False):
        if is_mapping:
            self.model.post_processing(coarse)

    def get_model_input(self, optimize_frames, is_mapping):
        batch_rays_d_list = []
        batch_rays_o_list = []
        batch_gt_depth_list = []
        batch_gt_color_list = []

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
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                get_samples(self.camera,
                            pixs_per_image,
                            c2w,
                            gt_depth,
                            gt_color,
                            device=self.model.device,
                            Hedge=Hedge,
                            Wedge=Wedge)
            batch_rays_o_list.append(batch_rays_o.float())
            batch_rays_d_list.append(batch_rays_d.float())
            batch_gt_depth_list.append(batch_gt_depth.float())
            batch_gt_color_list.append(batch_gt_color.float())

        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(
                -1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(
                -1)  # (N, 3, 1)
            t = (self.bounding_box.unsqueeze(0).to(self.device) -
                 det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth.squeeze(-1)  # (N)

        batch_rays_d = batch_rays_d[inside_mask]  # (N, 3)
        batch_rays_o = batch_rays_o[inside_mask]  # (N, 3)
        batch_gt_depth = batch_gt_depth[inside_mask]  # (N, 1)
        batch_gt_color = batch_gt_color[inside_mask]  # (N, 3)

        return {
            'rays_o': batch_rays_o,
            'rays_d': batch_rays_d,
            'target_s': batch_gt_color,
            'target_d': batch_gt_depth,
            'stage': self.stage,
        }

    def set_stage(self, is_mapping, step, n_iters, coarse=False):
        if not is_mapping:
            self.stage = 'color'
            return

        if self.model.config.coarse and coarse:
            self.stage = 'coarse'
        elif step <= self.config.mapping_middle_iter_ratio * n_iters:
            self.stage = 'middle'
        elif step <= self.config.mapping_fine_iter_ratio * n_iters:
            self.stage = 'fine'
        else:
            self.stage = 'color'

    def get_loss(self,
                 optimize_frames,
                 is_mapping,
                 step,
                 n_iters,
                 coarse=False):
        self.set_stage(is_mapping, step, n_iters, coarse=coarse)
        model_input = self.get_model_input(optimize_frames, is_mapping)
        model_outputs = self.model(model_input)
        loss_dict = self.model.get_loss_dict(model_outputs, model_input,
                                             is_mapping, self.stage)
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
                    'stage': 'color',
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
