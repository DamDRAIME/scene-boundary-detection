from enum import StrEnum, auto
from pathlib import Path
from typing import Callable

import torch


class Plot(StrEnum):
    ALIGNMENT = auto()
    COUNTOUR = auto()
    TWO_WAY = auto()
    THREE_WAY = auto()


class Window(StrEnum):
    ITAKURA = auto()
    SAKOECHIBA = auto()
    SLANTED_BAND = auto()


class DTWOutput:
    def __init__(self):
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.distance_fn: Callable
        self.distance: float
        self.distance_normalized: float
        self.index
        self.path
        self.distance_matrix

    def plot(self, dst_filepath: Path, type: Plot) -> Path:
        pass


def dtw(
    x: torch.Tensor,
    y: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], float],
    step_pattern,
    window: Window | None = None,
) -> DTWOutput:
    pass
