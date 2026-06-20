from dataclasses import dataclass, field
from typing import NewType

import numpy as np

from sbd.shared.models import Timestamps

SpriteSheetImg = NewType("SpriteSheetImg", np.ndarray)
SpriteImg = NewType("SpriteImg", np.ndarray)


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


@dataclass
class Sprite:
    idx: int
    local_idx: int
    sprite_sheet: SpriteSheet
    timestamp: Timestamps = field(init=False)

    def __post_init__(self):
        n_sprites = self.sprite_sheet.n_sprites
        duration = self.sprite_sheet.timestamp.duration / n_sprites
        start = self.sprite_sheet.timestamp.start + (duration * self.local_idx)
        end = self.sprite_sheet.timestamp.end if self.local_idx == n_sprites - 1 else start + duration
        self.timestamp = Timestamps(start, end)

    @property
    def filename(self) -> str:
        return f"sprite_sheet_{self.sprite_sheet.idx}_sprite_{self.local_idx}_{self.idx}.{self.sprite_sheet.type.split('/')[-1]}"
