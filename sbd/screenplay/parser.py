from __future__ import annotations

import json
from pathlib import Path

from sbd.common.utils.counter import Counter
from sbd.common.utils.detect_encoding import detect_encoding
from sbd.screenplay import utils as sp_utils
from sbd.screenplay.exceptions import SCREENPLAYParsingError
from sbd.screenplay.models import Label, Metadata, ParsedLine, Scene, ScreenplayElement

LABELS_TO_IGNORE = frozenset({Label.I, Label.G})
LABELS_PRIORITY = [Label.S, Label.C, Label.U, Label.P, Label.E, Label.N, Label.T, Label.M, Label.D, Label.O]
LABELS_PRESERVING_CONTINUITY_AFTER_O = [Label.S, Label.U, Label.P, Label.C]


class ScreenplayParser:
    def __init__(self, filepath: Path | str):
        self.filepath = Path(filepath)
        self.encoding = detect_encoding(self.filepath)
        self.meta: Metadata = None
        self.scenes: list[Scene] = []

    @property
    def data(self) -> dict:
        return {
            "meta": self.meta.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def parse(self) -> None:
        self.meta: Metadata = None
        self.scenes: list[Scene] = []

        raw_lines = self.filepath.read_text(encoding=self.encoding).splitlines()
        parsed_lines = self._parse_lines(raw_lines, LABELS_TO_IGNORE)

        id_generator = Counter()

        # ── Metadata content (before first non M line) ────────────────────────────────────────────────────────────────
        all_labels_except_M_O = [l for l in Label if l not in [Label.M, Label.O]]
        first_non_meta_line_idx = self._find_first_occurrence(parsed_lines, all_labels_except_M_O)

        metadata_buffer: list[ParsedLine] = []
        for pl in parsed_lines[:first_non_meta_line_idx]:
            if Label.M not in pl.labels:
                continue
            metadata_buffer.append(pl)
        if metadata_buffer:
            self.meta = Metadata(
                id=id_generator.next(),
                value="\n".join(pl.sanitized_content for pl in metadata_buffer),
                source_line_start_idx=metadata_buffer[0].line_idx,
                source_line_stop_idx=metadata_buffer[-1].line_idx,
                recently_modified=any(pl.is_stared for pl in metadata_buffer),
            )

        # ── Scenes ────────────────────────────────────────────────────────────────────────────────────────────────────
        act: str = ""
        scene_heading: list[ParsedLine] = []
        previous_label: Label = None
        scene_content: list[ScreenplayElement] = []
        lines_buffer: list[ParsedLine] = []

        def flush_buffer(accumulator: list[ScreenplayElement]) -> None:
            nonlocal lines_buffer, id_generator
            if lines_buffer:
                accumulator.append(sp_utils.build_screenplay_element(lines_buffer, id_generator))
                lines_buffer = []

        def flush_scene(accumulator: list[ScreenplayElement]) -> None:
            nonlocal act, scene_heading, scene_content, id_generator
            if scene_heading or scene_content:
                accumulator.append(
                    sp_utils.build_scene(
                        content=scene_content,
                        id=id_generator,
                        heading=scene_heading,
                        act="Cold-open" if not accumulator and not act else act,
                    )
                )
                scene_content = []
                scene_heading = []

        for pl in parsed_lines[first_non_meta_line_idx:]:
            if pl.primary_label == Label.D:  # Automatically flush Omitted Scenes on each occurrence.
                flush_buffer(scene_content)
                flush_scene(self.scenes)
                scene_heading.append(pl)
                flush_scene(self.scenes)
                continue
            if pl.primary_label == Label.O:
                if previous_label is not None and previous_label not in LABELS_PRESERVING_CONTINUITY_AFTER_O:
                    flush_buffer(scene_content)
                continue
            if not pl.sanitized_content:
                raise SCREENPLAYParsingError(f"Empty line not labeled `O` at {self.filepath}:{pl.line_idx}")
            if previous_label == Label.C and pl.primary_label == Label.C:
                flush_buffer(scene_content)
            elif previous_label == Label.C and pl.primary_label in (Label.P, Label.E, Label.U):
                pass
            elif pl.primary_label != previous_label:
                flush_buffer(scene_content)
                if pl.primary_label == Label.A:
                    flush_scene(self.scenes)
                    act = pl.sanitized_content
                    previous_label = Label.A
                    continue
                if pl.primary_label == Label.S:
                    if previous_label != Label.A:
                        flush_scene(self.scenes)
                    scene_heading.append(pl)
                    previous_label = Label.S
                    continue
                previous_label = pl.primary_label

            lines_buffer.append(pl)

        flush_buffer(scene_content)
        flush_scene(self.scenes)

    @classmethod
    def read(cls, filepath: Path | str) -> "ScreenplayParser":
        _self = cls(filepath)
        _self.parse()
        return _self

    def save(self, output_filepath: Path | str | None = None) -> Path:
        if output_filepath is None:
            output_filepath = self.filepath.with_suffix(".json")
        output_filepath = Path(output_filepath)
        with output_filepath.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        return output_filepath

    def _parse_lines(self, raw_lines: list[str], labels_to_ignore: frozenset[Label]) -> list[ParsedLine]:
        # TODO: Try to make this a staticmethod while preserving context in error or delegate it
        parsed_lines = []
        for line_idx, raw in enumerate(raw_lines, start=1):
            if "|" not in raw:
                raise SCREENPLAYParsingError(f"No label found at {self.filepath}:{line_idx}")
            labels_str, content = raw.split("|", 1)
            parsed_lines.append(
                ParsedLine(line_idx, sp_utils.sanitize_labels(labels_str, labels_to_ignore), content, LABELS_PRIORITY)
            )
        return parsed_lines

    def _find_first_occurrence(self, parsed_lines: list[ParsedLine], labels: list[Label]) -> int:
        # TODO: Try to make this a staticmethod
        for parsed_line in parsed_lines:
            if any(label in parsed_line.labels for label in labels):
                return parsed_line.line_idx - 1
        raise SCREENPLAYParsingError(f"No label(s) {str(labels)} detected in {self.filepath}")
