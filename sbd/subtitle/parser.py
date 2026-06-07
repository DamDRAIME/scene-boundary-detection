import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from sbd.utils.detect_encoding import detect_encoding


@dataclass
class SubTitle:
    idx: int
    filepath: Path
    start: datetime
    end: datetime
    content: str


class Timestamps(NamedTuple):
    start: datetime
    end: datetime


class SRTParsingError(ValueError):
    """Error linked to the parsing of a SRT file"""


class SRTParser:
    timestamp_line_pattern = re.compile(r"^(?P<start>.*) --> (?P<end>.*)$")

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.encoding = detect_encoding(self.filepath)
        self.line_idx: int = 0
        self._next_line: str | None = None
        self.subtitles: list[SubTitle] = []

    def parse(self):
        with self.filepath.open("r", encoding=self.encoding) as fh:
            idx: int | None = None
            content: list[str] = []
            timestamps: Timestamps | None = None
            for self.line_idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    if content and idx is not None and timestamps is not None:
                        self.subtitles.append(
                            SubTitle(
                                idx=idx,
                                filepath=self.filepath,
                                start=timestamps.start,
                                end=timestamps.end,
                                content=" ".join(content),
                            )
                        )
                        idx = None
                        content = []
                        timestamps = None

                    continue
                if idx is None:
                    idx = self._parse_idx_line(line)
                elif timestamps is None:
                    timestamps = self._parse_timestamps_line(line)
                else:
                    content.append(line)
            if content and idx is not None and timestamps is not None:
                self.subtitles.append(
                    SubTitle(
                        idx=idx,
                        filepath=self.filepath,
                        start=timestamps.start,
                        end=timestamps.end,
                        content=" ".join(content),
                    )
                )

    def _parse_idx_line(self, line: str) -> int:
        try:
            return int(line.strip())
        except (ValueError, OverflowError):
            raise SRTParsingError(
                "Invalid subtitle number line at {filepath}:{line_idx}".format(
                    filepath=self.filepath, line_idx=self.line_idx
                ),
            )

    def _parse_timestamps_line(self, line: str) -> Timestamps:
        def decode_timestamp(timestamp: str) -> datetime:
            formats = ["%H:%M:%S,%f", "%H:%M:%S.%f", "%M:%S,%f", "%M:%S.%f"]
            for format in formats:
                try:
                    return datetime.strptime(timestamp, format)
                except ValueError:
                    continue
            raise ValueError()

        line = line.strip()
        m = self.timestamp_line_pattern.match(line)
        if not m:
            raise SRTParsingError(
                "Invalid timestamps line at {filepath}:{line_idx}".format(
                    filepath=self.filepath, line_idx=self.line_idx
                ),
            )
        timestamps = []
        for start_end in ["start", "end"]:
            try:
                timestamps.append(decode_timestamp(m.group(start_end)))
            except ValueError:
                raise SRTParsingError(
                    "Invalid {start_end} timestamp at {filepath}:{line_idx}".format(
                        start_end=start_end, filepath=self.filepath, line_idx=self.line_idx
                    ),
                )
        return Timestamps(*timestamps)

    @classmethod
    def read(cls, filepath: Path) -> "SRTParser":
        _self = cls(filepath)
        _self.parse()
        return _self
