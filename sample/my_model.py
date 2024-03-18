"""slam/models/my_model.py."""

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from slam.common.camera import Camera
from slam.models.base_model import Model, ModelConfig


@dataclass
class MyModelConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: MyModel)
    # model config params


class MyModel(Model):
    """Model class."""

    config: MyModelConfig

    def __init__(
        self,
        config: MyModelConfig,
        camera: Camera,
        **kwargs,
    ) -> None:
        super().__init__(config=config, camera=camera, **kwargs)

    # inherit and implement the needed functions from Model
    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()

    def get_loss_dict(self,
                      outputs,
                      inputs,
                      is_mapping,
                      stage=None) -> Dict[str, torch.Tensor]:
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        pass

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        pass
