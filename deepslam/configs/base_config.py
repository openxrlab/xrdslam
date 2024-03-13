"""Base Configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Type


# Pretty printing class
class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function."""
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = '['
                for item in val:
                    flattened_val += str(item) + '\n'
                flattened_val = flattened_val.rstrip('\n')
                val = flattened_val + ']'
            lines += f'{key}: {str(val)}'.split('\n')
        return '\n    '.join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target
    attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
