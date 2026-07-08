from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple, Optional, Self

from sbd.common.models import Timestamps


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

    def serialize(self) -> dict:
        return {
            "idx": self.idx,
            "filepath": str(self.filepath),
            "line_idx": self.line_idx,
            "timestamp_start": self.timestamp.start.total_seconds(),
            "timestamp_end": self.timestamp.end.total_seconds(),
            "content": self.content,
            "x1": self.coordinates.x1 if self.coordinates else None,
            "y1": self.coordinates.y1 if self.coordinates else None,
            "x2": self.coordinates.x2 if self.coordinates else None,
            "y2": self.coordinates.y2 if self.coordinates else None,
        }

    @classmethod
    def deserialize(cls, obj: dict) -> Self:
        coordinates = None
        if obj["x1"] is not None:
            coordinates = Coordinates(obj["x1"], obj["y1"], obj["x2"], obj["y2"])
        return cls(
            idx=obj["idx"],
            filepath=Path(obj["filepath"]),
            line_idx=obj["line_idx"],
            timestamp=Timestamps(timedelta(seconds=obj["timestamp_start"]), timedelta(seconds=obj["timestamp_end"])),
            content=obj["content"],
            coordinates=coordinates,
        )


@dataclass
class Utterance:
    idx: int
    timestamp: Timestamps
    content: str
    # Assumes all associated subtitles come from the same file, since only `subtitles[0].filepath` is kept.
    filepath: Optional[Path] = None
    line_idxs: list[int] = field(default_factory=list)
    # Not persisted in UtteranceDataset Parquet files; empty when loaded back from disk.
    subtitles: list[SubTitle] = field(default_factory=list)

    @classmethod
    def from_subtitles(cls, *subtitles: SubTitle, idx: int) -> Self:
        return cls(
            idx=idx,
            content=" ".join([s.content for s in subtitles]),
            timestamp=Timestamps(subtitles[0].timestamp.start, subtitles[-1].timestamp.end),
            filepath=subtitles[0].filepath,
            line_idxs=[s.line_idx for s in subtitles],
            subtitles=subtitles,
        )

    def serialize(self) -> dict:
        return {
            "idx": self.idx,
            "timestamp_start": self.timestamp.start.total_seconds(),
            "timestamp_end": self.timestamp.end.total_seconds(),
            "content": self.content,
            "filepath": str(self.filepath) if self.filepath is not None else None,
            "line_idxs": list(self.line_idxs),
        }

    @classmethod
    def deserialize(cls, obj: dict) -> Self:
        return cls(
            idx=obj["idx"],
            timestamp=Timestamps(timedelta(seconds=obj["timestamp_start"]), timedelta(seconds=obj["timestamp_end"])),
            content=obj["content"],
            filepath=Path(obj["filepath"]) if obj["filepath"] is not None else None,
            line_idxs=list(obj["line_idxs"]),
        )
