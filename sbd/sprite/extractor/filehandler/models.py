from dataclasses import dataclass
from datetime import timedelta
from enum import auto, StrEnum
from typing import NewType

import numpy as np

from sbd.shared.models import Timestamps

SpriteSheetImg = NewType("SpriteSheetImg", np.ndarray)
SpriteImg = NewType("SpriteImg", np.ndarray)


class ExtractionMethod(StrEnum):
    SELECT = auto()
    SEEK = auto()


@dataclass
class SourceMetadata:
    fps: float
    duration: timedelta
    n_sprites: int
    sprite_shape: tuple[int, int]  # (H x W)


@dataclass
class SpriteSheetDownloadInfo:
    cid: str
    location: str
    type: str


@dataclass
class SpriteSheet:
    idx: int
    timestamp: Timestamps
    cid: str
    grid_shape: tuple[int, int]
    location: str | None = None
    type: str | None = None

    def add_download_info(self, data: SpriteSheetDownloadInfo) -> None:
        self.location = data.location
        self.type = data.type

    @property
    def filename(self) -> str:
        return f"sprite_sheet_{self.idx}.{self.type.split('/')[-1]}"

    @property
    def n_sprites(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]
