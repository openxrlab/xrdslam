"""Some utility code for configs."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict

from rich.console import Console

CONSOLE = Console()

# pylint: disable=import-outside-toplevel


# cannot use mutable types directly within dataclass;
# abstracting default factory calls
def to_immutable_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict.

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: dict(d))


def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        markup_string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()
