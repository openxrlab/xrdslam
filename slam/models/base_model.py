"""Base Model implementation."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.configs.base_config import InstantiateConfig


@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: Model)


class Model(nn.Module):
    """Model class."""

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        camera: Camera,
        bounding_box=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.camera = camera
        self.bounding_box = bounding_box
        self.kwargs = kwargs
        self.populate_modules()  # populate the modules

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    @abstractmethod
    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    def forward(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(input)

    @abstractmethod
    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        pass

    @abstractmethod
    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        pass
