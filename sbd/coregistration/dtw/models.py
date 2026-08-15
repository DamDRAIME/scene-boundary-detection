from dataclasses import dataclass
from enum import StrEnum, auto

import torch

from sbd.coregistration.distance import ProximityMetric


class Window(StrEnum):
    ITAKURA = auto()
    SAKOECHIBA = auto()
    SLANTED_BAND = auto()


class Method(StrEnum):
    CLASSIC = auto()  # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
    SUBSEQUENCE = auto()  # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html


@dataclass
class DTWOutput:
    a: torch.Tensor
    b: torch.Tensor
    metric: ProximityMetric
    method: Method
    distance: float
    optimal_warping_path: torch.Tensor
    cost_matrix: torch.Tensor | None = None
    accumulated_cost_matrix: torch.Tensor | None = None
