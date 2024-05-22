from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
from yacs.config import CfgNode as CN

from slam.common.camera import Camera
from slam.model_components.neural_recon_components.models.neuralrecon import \
    NeuralRecon
from slam.models.base_model import Model, ModelConfig


@dataclass
class NeuConModelConfig(ModelConfig):
    """Configuration for model instantiation."""
    _target: Type = field(default_factory=lambda: NeuConModel)

    alpha: int = 0
    pretrained_path: Optional[Path] = None
    model_cfg: Optional[List] = None


class NeuConModel(Model):
    """Model class."""

    config: NeuConModelConfig

    def __init__(
        self,
        config: NeuConModelConfig,
        camera: Camera,
        bounding_box,
        **kwargs,
    ) -> None:
        super().__init__(config=config,
                         camera=camera,
                         bounding_box=bounding_box,
                         **kwargs)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()

        self.cfg = self.get_default_config()
        if self.config.model_cfg is not None:
            self.cfg.merge_from_list(self.config.model_cfg)

        self.recon_net = NeuralRecon(self.cfg).cuda().eval()
        self.recon_net = torch.nn.DataParallel(self.recon_net, device_ids=[0])

        if self.config.pretrained_path is not None:
            self.load_pretrain()

    def load_pretrain(self):
        state_dict = torch.load(self.config.pretrained_path)
        self.recon_net.load_state_dict(state_dict['model'], strict=False)

    def get_outputs(self, input) -> Dict[str, Union[torch.Tensor, List]]:
        outputs, _ = self.recon_net(input, save_mesh=self.cfg.SAVE_SCENE_MESH)
        return outputs

    def get_default_config(self):
        cfg = CN()
        cfg.SAVE_SCENE_MESH = True
        cfg.MODEL = CN()
        cfg.MODEL.N_VOX = [128, 224, 192]
        cfg.MODEL.VOXEL_SIZE = 0.04
        cfg.MODEL.THRESHOLDS = [0, 0, 0]
        cfg.MODEL.N_LAYER = 3
        cfg.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
        cfg.MODEL.TEST_NUM_SAMPLE = [32768, 131072]
        cfg.MODEL.LW = [1.0, 0.8, 0.64]
        cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
        cfg.MODEL.PIXEL_STD = [1., 1., 1.]
        cfg.MODEL.POS_WEIGHT = 1.0
        cfg.MODEL.BACKBONE2D = CN()
        cfg.MODEL.BACKBONE2D.ARC = 'fpn-mnas'
        cfg.MODEL.SPARSEREG = CN()
        cfg.MODEL.SPARSEREG.DROPOUT = False
        cfg.MODEL.FUSION = CN()
        cfg.MODEL.FUSION.FUSION_ON = False
        cfg.MODEL.FUSION.HIDDEN_DIM = 64
        cfg.MODEL.FUSION.AVERAGE = False
        cfg.MODEL.FUSION.FULL = False
        return cfg
