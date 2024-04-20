import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch

from slam.common.camera import Camera
from slam.common.common import keyframe_selection_overlap
from slam.configs.base_config import InstantiateConfig
from slam.configs.config_utils import to_immutable_dict
from slam.engine.optimizers import OptimizerConfig, Optimizers
from slam.models.base_model import ModelConfig


@dataclass
class AlgorithmConfig(InstantiateConfig):
    """Algorithm  Config."""
    _target: Type = field(default_factory=lambda: Algorithm)
    model: ModelConfig = ModelConfig()
    keyframe_selection_method: str = 'overlap'
    keyframe_use_ray_sample: bool = True
    tracking_n_iters: int = 10
    mapping_n_iters: int = 60
    mapping_first_n_iters: int = 200
    coarse: bool = False
    mapping_window_size: int = 5
    separate_LR: bool = False
    rot_rep: str = 'quat'
    retain_graph: bool = False
    optimizers: Dict[str, Any] = to_immutable_dict({
        'model': {
            'optimizer': OptimizerConfig(lr=1e-2),
        },
        'tracking_pose': {
            'optimizer': OptimizerConfig(lr=1e-2),
        },
        'mapping_pose': {
            'optimizer': OptimizerConfig(lr=1e-3),
        },
    })


class Algorithm():
    def __init__(self, config: AlgorithmConfig, camera: Camera,
                 device: str) -> None:
        self.config = config
        self.camera = camera
        self.initialized = False
        self.finished = False
        self.lock = torch.multiprocessing.RLock()

        self.gt_c2w_list = []
        self.gt_c2w_list_ori = [
        ]  # equal to  gt_c2w_list when use_relative_pose=False
        self.estimate_c2w_list = []
        self.keyframe_graph = []
        self.bundle_adjust = False  # optimize pose and model_params

    @abstractmethod
    def get_model_input(self, optimize_frames, is_mapping):
        pass

    @abstractmethod
    def get_loss(self,
                 optimize_frames,
                 is_mapping,
                 step=None,
                 n_iters=None,
                 coarse=False):
        pass

    @abstractmethod
    def pre_precessing(self, cur_frame, is_mapping):
        pass

    @abstractmethod
    def post_processing(self, step, is_mapping, optimizer=None, coarse=False):
        pass

    @abstractmethod
    def render_img(self, c2w, gt_depth=None, idx=None):
        return None, None

    @abstractmethod
    def update_mesh(self):
        pass

    @abstractmethod
    def get_mesh(self):
        return None

    @abstractmethod
    def get_cloud(self, c2w_np, gt_depth_np):
        return None

    @abstractmethod
    def optimizer_config_update(self, max_iters, coarse=False):
        pass

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def add_framepose(self, c2w, gt_c2w, gt_c2w_ori):
        with self.lock:
            self.estimate_c2w_list.append(c2w)
            self.gt_c2w_list.append(gt_c2w)
            self.gt_c2w_list_ori.append(gt_c2w_ori)

    def update_framepose(self, idx, c2w):
        with self.lock:
            self.estimate_c2w_list[idx] = c2w

    def get_estimate_c2w_list(self):
        with self.lock:
            return self.estimate_c2w_list

    def get_gt_c2w_list(self):
        with self.lock:
            return self.gt_c2w_list

    def get_gt_c2w_list_ori(self):
        with self.lock:
            return self.gt_c2w_list_ori

    def get_keyframes(self):
        with self.lock:
            return self.keyframe_graph

    def add_keyframe(self, keyframe):
        with self.lock:
            self.keyframe_graph.append(keyframe)

    def is_separate_LR(self):
        with self.lock:
            return self.config.separate_LR

    def get_rot_rep(self):
        with self.lock:
            return self.config.rot_rep

    def is_initialized(self):
        with self.lock:
            return self.initialized

    def set_initialized(self):
        with self.lock:
            self.initialized = True

    def is_finished(self):
        with self.lock:
            return self.finished

    def set_finished(self):
        with self.lock:
            self.finished = True

    def setup_optimizers(self,
                         n_iters,
                         optimize_frames,
                         is_mapping=True,
                         coarse=False) -> Optimizers:
        # update optimizer config by n_iters
        self.optimizer_config_update(n_iters, coarse)
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
            model_params = self.model.get_param_groups()
            if not self.bundle_adjust or len(optimize_frames) == 1:
                return Optimizers(optimizer_config, {**model_params})
            else:
                if self.config.separate_LR:
                    pose_params = {'mapping_pose_r': [], 'mapping_pose_t': []}
                else:
                    pose_params = {'mapping_pose': []}
                oldest_frame_id = optimize_frames[-1].fid  # curframe id
                for keyframe in optimize_frames:
                    if keyframe.fid < oldest_frame_id:
                        oldest_frame_id = keyframe.fid
                for keyframe in optimize_frames:
                    # the oldest frame should be fixed to avoid drifting
                    if keyframe.fid != oldest_frame_id:
                        if self.config.separate_LR:
                            pose_params['mapping_pose_r'].extend(
                                [keyframe.get_params()[0]])
                            pose_params['mapping_pose_t'].extend(
                                [keyframe.get_params()[1]])
                        else:
                            pose_params['mapping_pose'].extend(
                                keyframe.get_params())
                return Optimizers(optimizer_config, {
                    **pose_params,
                    **model_params
                })

    def do_tracking(self, cur_frame):
        optimize_frames = [cur_frame]
        return self.optimize_update(self.config.tracking_n_iters,
                                    optimize_frames,
                                    is_mapping=False)

    def do_mapping(self, cur_frame):
        if not self.is_initialized():
            mapping_n_iters = self.config.mapping_first_n_iters
        else:
            mapping_n_iters = self.config.mapping_n_iters

        # select optimize frames
        with torch.no_grad():
            optimize_frames = self.select_optimize_frames(
                cur_frame,
                keyframe_selection_method=self.config.keyframe_selection_method
            )
        # optimize keyframes_pose, model_params, update model params
        self.optimize_update(mapping_n_iters,
                             optimize_frames,
                             is_mapping=True,
                             coarse=False)

        if not self.is_initialized():
            self.set_initialized()

    def optimize_update(self,
                        n_iters,
                        optimize_frames,
                        is_mapping,
                        coarse=False):
        with self.lock:
            # update optimizers params
            self.pre_precessing(optimize_frames[-1], is_mapping)
            # setup optimizers
            optimizers = self.setup_optimizers(n_iters,
                                               optimize_frames,
                                               is_mapping,
                                               coarse=coarse)
            # get best c2w for tracking
            candidate_c2w = None
            current_min_loss = 10000000000.
            for step in range(n_iters):
                optimizers.zero_grad_all()
                loss = self.get_loss(optimize_frames,
                                     is_mapping,
                                     step,
                                     n_iters,
                                     coarse=coarse)
                if not is_mapping and loss.cpu().item() < current_min_loss:
                    current_min_loss = loss.cpu().item()
                    candidate_c2w = optimize_frames[-1].get_pose().detach(
                    ).clone().cpu().numpy()
                loss.backward(
                    retain_graph=(self.config.retain_graph and is_mapping))
                self.post_processing(step,
                                     is_mapping,
                                     optimizers.optimizers,
                                     coarse=coarse)
                optimizers.optimizer_step_all(step=step)
                optimizers.scheduler_step_all()
            # return best c2w by min_loss
            return candidate_c2w

    def select_optimize_frames(self, cur_frame, keyframe_selection_method):
        optimize_frame = []
        window_size = self.config.mapping_window_size
        if len(self.keyframe_graph) <= window_size:
            optimize_frame = self.keyframe_graph[:]  # shallow copy
        elif keyframe_selection_method == 'random':
            optimize_frame = random.sample(self.keyframe_graph[:-1],
                                           (window_size - 2))
            # add last keyframe
            optimize_frame += [self.keyframe_graph[-1]]  # shallow copy
        elif keyframe_selection_method == 'overlap':
            optimize_frame = keyframe_selection_overlap(
                camera=self.camera,
                cur_frame=cur_frame,
                keyframes_graph=self.keyframe_graph[:-1],
                k=window_size - 2,
                use_ray_sample=self.config.keyframe_use_ray_sample,
                device=self.device)  # shallow copy
            # add last keyframe
            optimize_frame += [self.keyframe_graph[-1]]
        elif keyframe_selection_method == 'all':
            optimize_frame = self.keyframe_graph.copy()  # shallow copy
        # add current keyframe
        if cur_frame is not None:
            optimize_frame += [cur_frame]
        return optimize_frame
