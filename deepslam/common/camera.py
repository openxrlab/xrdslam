from dataclasses import dataclass


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
