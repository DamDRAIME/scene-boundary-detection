import re
from pathlib import Path
from typing import Optional

from sbd.subtitle import utils
from sbd.subtitle.exceptions import SRTParsingError
from sbd.subtitle.models import Timestamps, Coordinates, SubTitle
from sbd.shared.utils.detect_encoding import detect_encoding
from sbd.shared.utils.timedelta import timedelta_parse


class SubtitleParser:
    timestamp_line_pattern = re.compile(
        r"^(?P<start>[\d:,.]+)"  # Start timestamp
        r"\s*-->\s*"  # Timestamps separator
        r"(?P<end>[\d:,.]+)"  # End timestamp
        r"(?:\s+X1:(?P<x1>\d+))?(?:\s+X2:(?P<x2>\d+))?(?:\s+Y1:(?P<y1>\d+))?(?:\s+Y2:(?P<y2>\d+))?$"  # Coordinates
    )

    def __init__(self, filepath: Path | str, remove_html_tags: bool = True):
        self.filepath = Path(filepath)
        self.remove_html_tags = remove_html_tags
        self.encoding = detect_encoding(self.filepath)
        self.line_idx: int = 0
        self._next_line: str | None = None
        self.subtitles: list[SubTitle] = []

    def parse(self):
        self.subtitles = []  # Reset for the method to be idempotent
        with self.filepath.open("r", encoding=self.encoding) as fh:
            idx: int | None = None
            timestamps: Optional[Timestamps] = None
            coordinates: Optional[Coordinates] = None
            content: list[str] = []
            subtitle_header_line_idx: int = 1
            for self.line_idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    if content and idx is not None and timestamps is not None:
                        self._flush(idx, subtitle_header_line_idx, timestamps, content, coordinates)
                        idx, timestamps, content = None, None, []
                    continue

                if idx is None:
                    idx = self._parse_idx_line(line)
                    subtitle_header_line_idx = self.line_idx
                elif timestamps is None:
                    timestamps, coordinates = self._parse_timestamps_line(line)
                else:
                    content.append(self._parse_content_line(line))
            if content and idx is not None and timestamps is not None:
                self._flush(idx, subtitle_header_line_idx, timestamps, content, coordinates)

    @classmethod
    def read(cls, filepath: Path | str, remove_html_tags: bool = True) -> "SubtitleParser":
        _self = cls(filepath, remove_html_tags)
        _self.parse()
        return _self

    def _parse_idx_line(self, line: str) -> int:
        try:
            return int(line.strip())
        except (ValueError, OverflowError):
            raise SRTParsingError(
                "Invalid subtitle number line at {filepath}:{line_idx}".format(
                    filepath=self.filepath, line_idx=self.line_idx
                ),
            )

    def _parse_timestamps_line(self, line: str) -> tuple[Timestamps, Optional[Coordinates]]:
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
                timestamps.append(timedelta_parse(m.group(start_end)))
            except ValueError:
                raise SRTParsingError(
                    "Invalid {start_end} timestamp at {filepath}:{line_idx}".format(
                        start_end=start_end, filepath=self.filepath, line_idx=self.line_idx
                    ),
                )
        if m.group("x1") is None:
            return Timestamps(*timestamps), None
        # Assumes that if X1 is set then the other coordinates will be as well
        coords = Coordinates(*[int(m.group(coord)) for coord in ["x1", "y1", "x2", "y2"]])
        return Timestamps(*timestamps), coords

    def _parse_content_line(self, line: str) -> str:
        return line if not self.remove_html_tags else utils.remove_html_tags(line)

    def _flush(
        self,
        idx: int,
        subtitle_header_line_idx: int,
        timestamps: Timestamps,
        content: list[str],
        coordinates: Optional[Coordinates] = None,
    ) -> None:
        self.subtitles.append(
            SubTitle(
                idx=idx,
                filepath=self.filepath,
                line_idx=subtitle_header_line_idx,
                timestamp=timestamps,
                content=" ".join(content),
                coordinates=coordinates,
            )
        )
