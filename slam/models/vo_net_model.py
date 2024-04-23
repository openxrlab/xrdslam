from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import torch

from slam.common.camera import Camera
from slam.model_components.vonet_dpvo import VONet
from slam.models.base_model import Model, ModelConfig


@dataclass
class VONetModelConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: VONetModel)
    # model config params
    pretrained_path: Optional[Path] = None


class VONetModel(Model):
    """Model class."""

    config: VONetModelConfig

    def __init__(
        self,
        config: VONetModelConfig,
        camera: Camera,
        **kwargs,
    ) -> None:
        super().__init__(config=config, camera=camera, **kwargs)

    # inherit and implement the needed functions from Model
    def populate_modules(self):
        super().populate_modules()
        """Set the necessary modules to get the network working."""
        self.load_weights()

    def load_weights(self):
        """This function is from DPVO, licensed under the MIT License."""
        # load network from checkpoint file
        from collections import OrderedDict
        state_dict = torch.load(self.config.pretrained_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'update.lmbda' not in k:
                new_state_dict[k.replace('module.', '')] = v
        self.network = VONet()
        self.network.load_state_dict(new_state_dict)
        self.network.eval()
