from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional


class Timestamps(NamedTuple):
    start: datetime
    end: datetime


class Coordinates(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class SubTitle:
    idx: int
    filepath: Path
    line_idx: int
    start: datetime
    end: datetime
    content: str
    coordinates: Optional[Coordinates] = None


@dataclass
class SRTUtterance:
    idx: int
    start: datetime
    end: datetime
    content: str
    subtitles_indices: list[int]

    @classmethod
    def from_subtitles(cls, *subtitles: SubTitle, idx: int) -> "SRTUtterance":
        return cls(
            idx=idx,
            content=" ".join([s.content for s in subtitles]),
            start=subtitles[0].start,
            end=subtitles[-1].end,
            subtitles_indices=[s.idx for s in subtitles],
        )
