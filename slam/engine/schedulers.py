"""Scheduler Classes."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from slam.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config."""

    _target: Type = field(default_factory=lambda: Scheduler)
    """target class to instantiate"""


class Scheduler:
    """Base scheduler."""

    config: SchedulerConfig

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer,
                      lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """


@dataclass
class LRconfig:
    coarse: float = 0.0
    middle: float = 0.0
    fine: float = 0.0
    color: float = 0.005


@dataclass
class NiceSLAMSchedulerConfig(SchedulerConfig):
    """Config for SelfDefine schedule."""
    _target: Type = field(default_factory=lambda: NiceSLAMScheduler)
    coarse: bool = True
    middle_iter_ratio: float = 0.4
    fine_iter_ratio: float = 0.6
    stage_lr: LRconfig = LRconfig()
    max_steps: int = 1000


class NiceSLAMScheduler(Scheduler):

    config: NiceSLAMSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer,
                      lr_init: float) -> LRScheduler:
        def func(step):
            if self.config.coarse:
                learning_factor = self.config.stage_lr.coarse
            elif step <= self.config.max_steps * self.config.middle_iter_ratio:
                learning_factor = self.config.stage_lr.middle
            elif step <= self.config.max_steps * self.config.fine_iter_ratio:
                learning_factor = self.config.stage_lr.fine
            else:
                learning_factor = self.config.stage_lr.color
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


@dataclass
class PointSLAMSchedulerConfig(SchedulerConfig):
    """Config for SelfDefine schedule."""
    _target: Type = field(default_factory=lambda: PointSLAMScheduler)
    geo_iter_ratio: float = 0.4
    start_lr: float = 0.001
    end_lr: float = 0.005
    max_steps: int = 1000


class PointSLAMScheduler(Scheduler):

    config: PointSLAMSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer,
                      lr_init: float) -> LRScheduler:
        def func(step):
            if step <= self.config.max_steps * self.config.geo_iter_ratio:
                learning_factor = self.config.start_lr
            else:
                learning_factor = self.config.end_lr
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
