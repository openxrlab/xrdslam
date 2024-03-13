"""Optimizers class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from deepslam.configs.base_config import PrintableConfig


# Optimizer related configs
@dataclass
class OptimizerConfig(PrintableConfig):
    """Basic optimizer config with RAdam."""

    _target: Type = torch.optim.Adam
    """The optimizer class to use."""
    lr: float = 0.0005
    """The learning rate to use."""
    eps: float = 1e-08
    """The epsilon value to use."""
    betas: Tuple[float, float] = (0.9, 0.999)
    """The betas values to use."""
    max_norm: Optional[float] = None
    """The max norm to use for gradient clipping."""
    accum_step: Optional[int] = None

    # TODO: somehow make this more generic. i dont like the idea of overriding
    # the setup function but also not sure how to go about passing things into
    # predefined torch objects.
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop('_target')
        kwargs.pop('max_norm')
        kwargs.pop('accum_step')
        return self._target(params, **kwargs)


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam."""

    _target: Type = torch.optim.Adam
    weight_decay: float = 0
    """The weight decay to use."""


@dataclass
class RAdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with RAdam."""

    _target: Type = torch.optim.RAdam
    weight_decay: float = 0
    """The weight decay to use."""


class Optimizers:
    """A set of optimizers.

    Args:
        config: The optimizer configuration object.
        param_groups: A dictionary of parameter groups to optimize.
    """
    def __init__(self,
                 config: Dict[str, Any] = None,
                 param_groups: Dict[str, List[Parameter]] = None,
                 optimizers: Dict[str, Any] = None) -> None:
        if optimizers:
            self.config = config
            self.schedulers = {}
            self.optimizers = optimizers
        else:
            self.config = config
            self.optimizers = {}
            self.parameters = {}
            self.schedulers = {}
            for param_group_name, params in param_groups.items():
                if param_group_name not in config:
                    raise RuntimeError(
                        f"""Optimizer config for '{param_group_name}' not found
                        in config file. Make sure you specify an optimizer for
                        each parameter group. Provided configs were:
                        {config.keys()}""")
                lr_init = config[param_group_name]['optimizer'].lr
                self.optimizers[param_group_name] = config[param_group_name][
                    'optimizer'].setup(params=params)
                self.parameters[param_group_name] = params
                if config[param_group_name]['scheduler']:
                    self.schedulers[param_group_name] = (
                        config[param_group_name]
                        ['scheduler'].setup().get_scheduler(
                            optimizer=self.optimizers[param_group_name],
                            lr_init=lr_init))

    # only use for co-slam
    def __add__(self, other: 'Optimizers') -> 'Optimizers':
        """Override the addition operation."""
        optimizers = {**self.optimizers, **other.optimizers}
        configs = {**self.config, **other.config}
        return Optimizers(config=configs, optimizers=optimizers)

    def optimizer_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding optimizer.

        Args:
            param_group_name: name of optimizer to step forward
        """
        self.optimizers[param_group_name].step()

    def scheduler_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding scheduler.

        Args:
            param_group_name: name of scheduler to step forward
        """
        if 'scheduler' in self.config[param_group_name]:
            self.schedulers[param_group_name].step()

    def zero_grad_all(self) -> None:
        """Zero the gradients for all optimizer parameters."""
        for param_group, optimizer in self.optimizers.items():
            accum_step = self.config[param_group]['optimizer'].accum_step
            if accum_step is None:
                optimizer.zero_grad(set_to_none=True)

    def optimizer_scaler_step_all(self, grad_scaler: GradScaler) -> None:
        """Take an optimizer step using a grad scaler.

        Args:
            grad_scaler: GradScaler to use
        """
        for param_group, optimizer in self.optimizers.items():
            max_norm = self.config[param_group]['optimizer'].max_norm
            if max_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters[param_group],
                                               max_norm)
            if any(
                    any(p.grad is not None for p in g['params'])
                    for g in optimizer.param_groups):
                grad_scaler.step(optimizer)

    def optimizer_step_all(self, step: int) -> None:
        """Run step for all optimizers."""
        for param_group, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            max_norm = self.config[param_group]['optimizer'].max_norm
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters[param_group],
                                               max_norm)
            accum_step = self.config[param_group]['optimizer'].accum_step
            if accum_step is None:
                optimizer.step()
            elif (accum_step is not None and (step + 1) % accum_step == 0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    def scheduler_step_all(self) -> None:
        """Run step for all schedulers.

        Args:
            step: the current step
        """
        for param_group, scheduler in self.schedulers.items():
            scheduler.step()
            # lr = scheduler.get_last_lr()[0]
            # print(f"learning_rate/{param_group}: ", lr)

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the optimizer state from previous checkpoint.

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.optimizers[k].load_state_dict(v)

    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the scheduler state from previous checkpoint.

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.schedulers[k].load_state_dict(v)
