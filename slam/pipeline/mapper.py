from dataclasses import dataclass, field
from typing import Type

import torch

from slam.configs.base_config import InstantiateConfig


@dataclass
class MapperConfig(InstantiateConfig):
    """Mapper  Config."""
    _target: Type = field(default_factory=lambda: Mapper)
    keyframe_every: int = 5


class Mapper():
    def __init__(self, config: MapperConfig) -> None:
        self.config = config

    def spin(self, map_buffer, algorithm, event_ready, event_processed):
        cur_frame = None
        while True:

            if not map_buffer.empty():
                event_ready.wait()

                cur_frame = map_buffer.get()

                algorithm.do_mapping(cur_frame)
                # update pose
                algorithm.update_framepose(cur_frame.fid,
                                           cur_frame.get_pose().detach())

                if cur_frame.fid % self.config.keyframe_every == 0:
                    algorithm.add_keyframe(cur_frame)

                torch.cuda.empty_cache()

                event_ready.clear()
                event_processed.set()

            # exit
            if algorithm.is_finished() and map_buffer.empty():
                event_processed.clear()
                event_ready.clear()
                break
