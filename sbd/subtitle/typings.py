from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple, Optional


@dataclass
class Timestamps:
    start: timedelta
    end: timedelta

    @property
    def mid(self):
        return self.start + ((self.end - self.start) / 2)

    def first_half(self) -> "Timestamps":
        return Timestamps(self.start, self.mid)

    def second_half(self) -> "Timestamps":
        return Timestamps(self.mid, self.end)

    def __repr__(self) -> str:
        return f"{str(self.start)} --> {str(self.end)}"


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
    subtitles_indices: list[int]

    @classmethod
    def from_subtitles(cls, *subtitles: SubTitle, idx: int) -> "SRTUtterance":
        return cls(
            idx=idx,
            content=" ".join([s.content for s in subtitles]),
            timestamp=Timestamps(subtitles[0].timestamp.start, subtitles[-1].timestamp.end),
            subtitles_indices=[s.idx for s in subtitles],
        )
