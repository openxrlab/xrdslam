import functools
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch

from slam.algorithms.base_algorithm import Algorithm, AlgorithmConfig
from slam.common.camera import Camera
from slam.common.common import keyframe_selection_overlap, rgbd2pcd


@dataclass
class SplaTAMConfig(AlgorithmConfig):
    """SplaTAM  Config."""
    _target: Type = field(default_factory=lambda: SplaTAM)
    mapping_sil_thres: float = 0.5
    render_mode: str = 'color'  # ['color', 'depth' or 'centers']


class SplaTAM(Algorithm):

    config: SplaTAMConfig

    def __init__(self, config: SplaTAMConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        self.model = self.config.model.setup(camera=camera, bounding_box=None)
        self.model.to(device)
        self.bundle_adjust = False

    def select_optimize_frames(self, cur_frame, keyframe_selection_method):
        optimize_frame = []
        window_size = self.config.mapping_window_size
        if len(self.keyframe_graph) > 0:
            optimize_frame = keyframe_selection_overlap(
                camera=self.camera,
                cur_frame=cur_frame,
                keyframes_graph=self.keyframe_graph[:-1],
                k=window_size - 2,
                use_ray_sample=self.config.keyframe_use_ray_sample,
                device=self.device)  # shallow copy
            # add last keyframe
            optimize_frame += [self.keyframe_graph[-1]]
        # add current keyframe
        if cur_frame is not None:
            optimize_frame += [cur_frame]
        return optimize_frame

    def get_model_input(self, optimize_frames, is_mapping):
        # Randomly select a frame until current time step amongst keyframes
        rand_idx = np.random.randint(0, len(optimize_frames))

        iter_color = optimize_frames[rand_idx].rgb
        iter_depth = optimize_frames[rand_idx].depth
        c2w = optimize_frames[rand_idx].get_pose().to(self.device)
        iter_w2c = torch.inverse(c2w)
        retain_grad = True

        return {
            'w2c': iter_w2c,
            'target_s': iter_color,
            'target_d': iter_depth,
            'is_mapping': is_mapping,
            'retain_grad': retain_grad
        }

    def get_loss(self,
                 optimize_frames,
                 is_mapping,
                 step,
                 n_iters,
                 coarse=False):
        model_input = self.get_model_input(optimize_frames, is_mapping)
        model_outputs = self.model(model_input)
        loss_dict = self.model.get_loss_dict(model_outputs, model_input,
                                             is_mapping)
        loss = functools.reduce(torch.add, loss_dict.values())
        return loss

    def pre_precessing(self, cur_frame, is_mapping):
        if is_mapping:
            self.model.model_update(cur_frame)

    def post_processing(self, step, is_mapping, optimizer=None, coarse=False):
        if is_mapping:
            self.model.post_processing(step, optimizer)

    def render_img(self, c2w, gt_depth=None, idx=None, use_sil_depth=True):
        with self.lock and torch.no_grad():
            if isinstance(c2w, np.ndarray):
                c2w = torch.from_numpy(c2w).to(self.device)
            iter_w2c = torch.inverse(c2w)
            model_input = {
                'w2c': iter_w2c,
                'is_mapping': True,
                'retain_grad': False
            }
            model_outputs = self.model(model_input)
            rgb = model_outputs['rgb']  # [3, H, W]
            if not use_sil_depth:
                rdepth = model_outputs['depth']  # [1, H, W]
                rdepth = rdepth.squeeze(0)  # [H, W]
            else:
                depth_sil = model_outputs['depth_sil']
                rdepth = depth_sil[0, :, :]  # [H, W]
            valid_depth_mask = (gt_depth > 0)
            valid_depth_mask = torch.from_numpy(valid_depth_mask).to(
                self.device) & (~torch.isnan(rdepth))
            rdepth = rdepth * valid_depth_mask.to(self.device)
            rcolor = rgb * valid_depth_mask
            return rcolor.clone().detach().cpu().permute(
                1, 2, 0).numpy(), rdepth.clone().detach().cpu().numpy()

    def get_cloud(self, c2w_np, gt_depth_np):
        with self.lock and torch.no_grad():
            rcolor_np, rdepth_np = self.render_img(c2w_np,
                                                   gt_depth_np,
                                                   use_sil_depth=False)
            init_pts, init_cols = rgbd2pcd(rcolor_np,
                                           rdepth_np,
                                           c2w_np,
                                           self.camera,
                                           self.config.render_mode,
                                           device=self.device)
            return init_pts, init_cols
