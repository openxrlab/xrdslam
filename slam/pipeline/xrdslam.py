from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Optional, Type

import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from slam.algorithms.base_algorithm import AlgorithmConfig
from slam.common.datasets import get_dataset
from slam.configs.base_config import InstantiateConfig
from slam.pipeline.mapper import MapperConfig
from slam.pipeline.tracker import TrackerConfig
from slam.pipeline.visualizer import VisualizerConfig

torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class XRDSLAMConfig(InstantiateConfig):
    """XRDSLAM SLAM  Config."""
    _target: Type = field(default_factory=lambda: XRDSLAM)

    data: Optional[Path] = None
    data_type: Optional[str] = 'tum'
    out_dir: Path = Path('outputs')

    tracker: TrackerConfig = TrackerConfig()
    mapper: MapperConfig = MapperConfig()
    method: AlgorithmConfig = AlgorithmConfig()
    enable_vis: bool = True
    visualizer: VisualizerConfig = VisualizerConfig()

    device: str = 'cuda:0'


class XRDSLAM():
    def __init__(self, config: XRDSLAMConfig) -> None:
        self.config = config
        # get dataset
        self.dataset = get_dataset(config.data, self.config.data_type,
                                   self.config.device)
        self.camera = self.dataset.get_camera()
        # ShareAlgorithm
        mp.set_start_method('spawn', force=True)
        # Use a BaseManager to create a shared Algorithm instance
        manager = BaseManager()
        manager.register('ShareAlgorithm', self.config.method._target)
        manager.start()
        self.method = manager.ShareAlgorithm(config=self.config.method,
                                             camera=self.camera,
                                             device=self.config.device)

        # mapframe buffer
        self.map_buffer = mp.Queue(maxsize=1)
        # visualize buffer
        self.viz_buffer = mp.Queue(maxsize=10)

        # strict sync
        self.event_ready = mp.Event()
        self.event_processed = mp.Event()

        self.tracker = self.config.tracker.setup(
            dataset=self.dataset,
            enable_vis=self.config.enable_vis,
            out_dir=self.config.out_dir)
        self.mapper = self.config.mapper.setup()
        if self.config.enable_vis:
            self.visualizer = self.config.visualizer.setup(
                camera=self.camera, out_dir=self.config.out_dir)

    def run(self):
        mapping_process = mp.Process(target=self.mapper.spin,
                                     args=(self.map_buffer, self.method,
                                           self.event_ready,
                                           self.event_processed))
        mapping_process.start()
        tracking_process = mp.Process(target=self.tracker.spin,
                                      args=(self.map_buffer, self.method,
                                            self.viz_buffer, self.event_ready,
                                            self.event_processed))
        tracking_process.start()
        self.processes = [tracking_process, mapping_process]

        if self.config.enable_vis:
            vis_process = mp.Process(target=self.visualizer.spin,
                                     args=(self.viz_buffer, ))
            vis_process.start()
            self.processes += [vis_process]

        for p in self.processes:
            p.join()
