from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Self

from sbd.shared.models import Timestamps


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
    timestamp: Timestamps
    content: str
    coordinates: Optional[Coordinates] = None


@dataclass
class SRTUtterance:
    idx: int
    timestamp: Timestamps
    content: str
    subtitles: list[SubTitle]

    @classmethod
    def from_subtitles(cls, *subtitles: SubTitle, idx: int) -> Self:
        return cls(
            idx=idx,
            content=" ".join([s.content for s in subtitles]),
            timestamp=Timestamps(subtitles[0].timestamp.start, subtitles[-1].timestamp.end),
            subtitles=subtitles,
        )
