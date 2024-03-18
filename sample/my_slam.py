"""slam/algorithms/my_slam.py."""
import functools
from dataclasses import dataclass, field
from typing import Type

import torch

from slam.algorithms.base_algorithm import Algorithm, AlgorithmConfig
from slam.common.camera import Camera


@dataclass
class MySLAMConfig(AlgorithmConfig):
    """MyAlgorithm  Config."""
    _target: Type = field(default_factory=lambda: MySLAM)
    # algorithm config params


class MySLAM(Algorithm):

    config: MySLAMConfig

    def __init__(self, config: MySLAMConfig, camera: Camera,
                 device: str) -> None:
        super().__init__(config, camera, device)
        # setup model
        self.model = self.config.model.setup(camera=camera)
        self.model.to(device)

    # inherit and implement the needed functions from Algorithm
    def get_model_input(self, optimize_frames, is_mapping):
        pass

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
        return None, None

    # Note: temporary use
    def optimize_update(self,
                        n_iters,
                        optimize_frames,
                        is_mapping,
                        coarse=False):
        return optimize_frames[-1].get_pose().detach().clone().cpu().numpy()
