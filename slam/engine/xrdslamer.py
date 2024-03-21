from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import yaml
from rich.console import Console

from slam.configs.base_config import InstantiateConfig
from slam.pipeline.xrdslam import XRDSLAM, XRDSLAMConfig

CONSOLE = Console(width=120)


@dataclass
class XRDSLAMerConfig(InstantiateConfig):
    """Configuration for xrdslam regimen."""

    _target: Type = field(default_factory=lambda: XRDSLAMer)

    xrdslam: XRDSLAMConfig = XRDSLAMConfig()

    data: Optional[Path] = None
    data_type: Optional[str] = 'tumrgbd'
    output_dir: Path = Path('outputs')
    algorithm_name: Optional[str] = None

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal."""
        CONSOLE.rule('Config')
        CONSOLE.print(self)
        CONSOLE.rule('')

    def save_config(self) -> None:
        """Save config to base directory."""
        base_dir = self.output_dir
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / 'config.yml'
        CONSOLE.log(f'Saving config to: {config_yaml_path}')
        config_yaml_path.write_text(yaml.dump(self), 'utf8')


class XRDSLAMer:

    slam: XRDSLAM

    def __init__(self, config: XRDSLAMerConfig) -> None:
        self.config = config

    def setup(self):
        self.config.xrdslam.out_dir = self.config.output_dir
        self.slam = self.config.xrdslam.setup()

    def run(self) -> None:
        self.slam.run()
