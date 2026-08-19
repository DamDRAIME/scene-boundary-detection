from dataclasses import dataclass
from datetime import timedelta
from typing import NewType

import numpy as np

FrameImg = NewType("FrameImg", np.ndarray)


@dataclass
class Shot:
    timestamp: timedelta
    frame: FrameImg
    score: float | None = None


@dataclass
class SourceMetadata:
    fps: float
    duration: timedelta
    frame_shape: tuple[int, int]  # (H x W)
