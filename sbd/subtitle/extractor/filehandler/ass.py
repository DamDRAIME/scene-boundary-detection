import re
from pathlib import Path
from typing import Iterator, Optional

from sbd.common.utils.detect_encoding import detect_encoding
from sbd.common.utils.timedelta import timedelta_parse
from sbd.exceptions import ASSParsingError
from sbd.subtitle.extractor.filehandler import utils
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.models import SubTitle, Timestamps


class ASSFileHandler(SubtitleFileHandler):
    file_suffix = ".ass"
    section_header_pattern = re.compile(r"^\[(?P<section>.+)\]$")
    line_break_pattern = re.compile(r"\\[Nnh]")

    def __init__(self, filepath: Path | str, remove_override_blocks: bool = True):
        super().__init__(filepath)
        self.remove_override_blocks = remove_override_blocks
        self.encoding = detect_encoding(self.filepath)
        self.line_idx: int = 0

    def iter_subtitles(self) -> Iterator[SubTitle]:
        with self.filepath.open("r", encoding=self.encoding) as fh:
            in_events_section = False
            fields: Optional[list[str]] = None
            idx = 0
            for self.line_idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                section_match = self.section_header_pattern.match(line)
                if section_match:
                    in_events_section = section_match.group("section").strip().lower() == "events"
                    continue
                if not in_events_section:
                    continue

                if line.lower().startswith("format:"):
                    fields = self._parse_format_line(line)
                    continue

                if not line.lower().startswith("dialogue:"):
                    continue
                if fields is None:
                    raise ASSParsingError(
                        "Dialogue line encountered before Format line at {filepath}:{line_idx}".format(
                            filepath=self.filepath, line_idx=self.line_idx
                        ),
                    )

                idx += 1
                yield self._parse_dialogue_line(line, idx, fields)

    def _parse_format_line(self, line: str) -> list[str]:
        fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
        for required in ("start", "end", "text"):
            if required not in fields:
                raise ASSParsingError(
                    "Missing '{required}' field in Format line at {filepath}:{line_idx}".format(
                        required=required, filepath=self.filepath, line_idx=self.line_idx
                    ),
                )
        return fields

    def _parse_dialogue_line(self, line: str, idx: int, fields: list[str]) -> SubTitle:
        values = line.split(":", 1)[1].split(",", maxsplit=len(fields) - 1)
        if len(values) != len(fields):
            raise ASSParsingError(
                "Invalid Dialogue line at {filepath}:{line_idx}".format(filepath=self.filepath, line_idx=self.line_idx),
            )
        row = dict(zip(fields, values))

        timestamps = self._parse_timestamps(row["start"], row["end"])
        content = self._parse_content_field(row["text"])
        return SubTitle(
            idx=idx,
            filepath=self.filepath,
            line_idx=self.line_idx,
            timestamp=timestamps,
            content=content,
        )

    def _parse_timestamps(self, start: str, end: str) -> Timestamps:
        try:
            return Timestamps(timedelta_parse(start.strip()), timedelta_parse(end.strip()))
        except ValueError:
            raise ASSParsingError(
                "Invalid timestamps at {filepath}:{line_idx}".format(filepath=self.filepath, line_idx=self.line_idx),
            )

    def _parse_content_field(self, text: str) -> str:
        text = self.line_break_pattern.sub(" ", text)
        if self.remove_override_blocks:
            text = utils.remove_ass_override_blocks(text)
        return " ".join(text.split())
