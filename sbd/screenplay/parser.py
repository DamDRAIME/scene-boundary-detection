"""
Screenplay parser: converts annotated .screenplay files into structured JSON.

Label reference
---------------
C  Character      who is speaking (dialogue block header)
D  Deletion       omitted/deleted scene marker  ← treated as a scene boundary
E  Extension      how/where the voice is heard (often embedded in C line)
G  Camera         camera direction (ignored; absorbed into N)
I  Introduction   character intro (ignored; absorbed into N)
M  Metadata       title, author notes, production info
N  Narrative      action / description
O  Omit           blank lines, page headers, CONTINUED markers (skipped)
P  Parenthetical  delivery note inside a dialogue block
S  Slugline       scene heading
T  Transition     CUT TO, FADE IN, etc.
U  Utterance      spoken dialogue
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Self

from sbd.screenplay.exceptions import SCREENPLAYParsingError
from sbd.screenplay.models import (
    Deletion,
    Dialogue,
    Metadata,
    Narrative,
    Parenthetical,
    ParsedLine,
    Scene,
    Transition,
    Utterance,
)
from sbd.screenplay.typings import Label, PreludeContent, SceneContent
from sbd.screenplay.utils import extract_primary_label, sanitize_labels
from sbd.shared.utils import detect_encoding
from sbd.shared.utils.counter import Counter

LABELS_TO_IGNORE = frozenset({Label.I, Label.G})
LABELS_PRIORITY = [Label.S, Label.C, Label.U, Label.P, Label.E, Label.N, Label.T, Label.M, Label.D, Label.O]
LABELS_PRESERVING_CONTINUITY_AFTER_O = [Label.S, Label.U, Label.P]


# ── Character-line parsing ───────────────────────────────────────────────────


def _extract_name_ext(text: str) -> tuple[str, list[str]]:
    exts = re.findall(r"\(([^)]+)\)", text)
    name = re.sub(r"\s*\([^)]+\)", "", text).strip().rstrip(".")
    return name, exts


def _parse_char_line(content: str) -> list[tuple[str, list[str]]]:
    """
    Parse a character-header line; return [(name, extensions)].

    Length 1 for a single speaker, length 2 for simultaneous speakers.
    Simultaneity is detected by finding two name clusters separated by a gap
    of more than 6 characters.
    """
    stripped = content.rstrip()
    tokens = list(re.finditer(r"\S+", stripped))
    if not tokens:
        return []

    groups: list[list] = []
    cur = [tokens[0]]
    for tok in tokens[1:]:
        if tok.start() - cur[-1].end() <= 3:
            cur.append(tok)
        else:
            groups.append(cur)
            cur = [tok]
    groups.append(cur)

    def _span(g: list) -> str:
        return stripped[g[0].start() : g[-1].end()]

    if len(groups) == 1:
        return [_extract_name_ext(_span(groups[0]))]

    gaps = [(groups[i + 1][0].start() - groups[i][-1].end(), i) for i in range(len(groups) - 1)]
    max_gap, max_idx = max(gaps)

    if max_gap <= 6:
        return [_extract_name_ext(stripped[groups[0][0].start() : groups[-1][-1].end()])]

    left_g, right_g = groups[: max_idx + 1], groups[max_idx + 1 :]
    return [
        _extract_name_ext(stripped[left_g[0][0].start() : left_g[-1][-1].end()]),
        _extract_name_ext(stripped[right_g[0][0].start() : right_g[-1][-1].end()]),
    ]


# ── Simultaneous-column splitter ─────────────────────────────────────────────


def _split_sim_content(
    content: str,
    ref_split: int | None,
) -> tuple[str, str, int | None]:
    """
    Split a U/P line for simultaneous dialogue into (left_text, right_text, new_ref_split).

    Two speakers' text sits in horizontal columns separated by ≥ 3 spaces.
    When only one column has content, ref_split (start of the right column from
    a previous two-column line) decides which speaker it belongs to.
    """
    stripped = content.rstrip()
    tokens = list(re.finditer(r"\S+", stripped))
    if not tokens:
        return "", "", ref_split

    groups: list[list] = []
    cur = [tokens[0]]
    for tok in tokens[1:]:
        if tok.start() - cur[-1].end() <= 2:
            cur.append(tok)
        else:
            groups.append(cur)
            cur = [tok]
    groups.append(cur)

    if len(groups) >= 2:
        gaps = [(groups[i + 1][0].start() - groups[i][-1].end(), i) for i in range(len(groups) - 1)]
        max_gap, max_idx = max(gaps)
        if max_gap >= 3:
            left_end = groups[max_idx][-1].end()
            right_start = groups[max_idx + 1][0].start()
            return content[:left_end].strip(), content[right_start:].strip(), right_start

    all_text = content.strip()
    if ref_split is not None:
        if tokens[0].start() >= ref_split:
            return "", all_text, ref_split
        return all_text, "", ref_split
    return all_text, "", None


def _is_parenthetical(text: str) -> bool:
    return text.strip().startswith("(")


def _clean_parenthetical(text: str) -> str:
    t = text.strip()
    return t[1:-1].strip() if t.startswith("(") and t.endswith(")") else t


# ── Scene content builder ────────────────────────────────────────────────────


def _build_content(lines: list[ParsedLine], ids: Counter) -> list[SceneContent]:
    result: list[SceneContent] = []
    # Narrative buffer: list of (text, line_num) so we can record the span.
    narrative_buf: list[tuple[str, int]] = []
    dial_left: Dialogue | None = None
    dial_right: Dialogue | None = None
    split_pos: int | None = None

    def flush_narrative() -> None:
        nonlocal narrative_buf
        if narrative_buf:
            result.append(
                Narrative(
                    id=ids.next(),
                    value=" ".join(t for t, _ in narrative_buf),
                    line_start=narrative_buf[0][1],
                    line_stop=narrative_buf[-1][1],
                )
            )
            narrative_buf = []

    def flush_dialogues() -> None:
        nonlocal dial_left, dial_right, split_pos
        if dial_left is not None:
            result.append(dial_left)
        if dial_right is not None:
            result.append(dial_right)
        dial_left = dial_right = split_pos = None

    def append_to_dialogue(dlg: Dialogue, text: str, is_paren: bool, line_num: int) -> None:
        if not text:
            return
        if is_paren:
            dlg.content.append(
                Parenthetical(
                    id=ids.next(),
                    value=_clean_parenthetical(text),
                    line_start=line_num,
                    line_stop=line_num,
                )
            )
        else:
            dlg.content.append(
                Utterance(
                    id=ids.next(),
                    value=text,
                    line_start=line_num,
                    line_stop=line_num,
                )
            )
        dlg.line_stop = line_num  # extend to last utterance/parenthetical

    for labels, content, line_num in lines:
        primary_label = extract_primary_label(labels, LABELS_PRIORITY)

        if primary_label == Label.O:
            # Ends a narrative paragraph but leaves any open dialogue block intact
            # (page-continuation markers appear inside dialogue sections).
            flush_narrative()
            continue

        text = content.strip()
        if not text:
            continue

        if primary_label == Label.S:
            flush_narrative()
            flush_dialogues()
            continue

        if primary_label == Label.N:
            if dial_left is not None:
                flush_dialogues()
            narrative_buf.append((text, line_num))
            continue

        if primary_label == Label.T:
            flush_narrative()
            flush_dialogues()
            result.append(Transition(id=ids.next(), value=text, line_start=line_num, line_stop=line_num))
            continue

        if primary_label == Label.M:
            flush_narrative()
            flush_dialogues()
            result.append(Metadata(id=ids.next(), value=text, line_start=line_num, line_stop=line_num))
            continue

        if primary_label == Label.D:
            flush_narrative()
            flush_dialogues()
            result.append(Deletion(id=ids.next(), value=text, line_start=line_num, line_stop=line_num))
            continue

        if primary_label == Label.C:
            flush_narrative()
            flush_dialogues()
            chars = _parse_char_line(content)
            if not chars:
                continue
            if len(chars) == 1:
                name, exts = chars[0]
                dial_left = Dialogue(
                    id=ids.next(),
                    character=name,
                    extensions=exts,
                    line_start=line_num,
                    line_stop=line_num,
                )
                dial_right = None
                split_pos = None
            else:
                name1, exts1 = chars[0]
                name2, exts2 = chars[1]
                id1, id2 = ids.next(), ids.next()
                dial_left = Dialogue(
                    id=id1,
                    character=name1,
                    extensions=exts1,
                    same_time_as=id2,
                    line_start=line_num,
                    line_stop=line_num,
                )
                dial_right = Dialogue(
                    id=id2,
                    character=name2,
                    extensions=exts2,
                    same_time_as=id1,
                    line_start=line_num,
                    line_stop=line_num,
                )
                split_pos = None
            continue

        if primary_label in (Label.U, Label.P):
            if dial_left is None:
                continue
            if dial_right is not None:
                l_text, r_text, split_pos = _split_sim_content(content, split_pos)
                append_to_dialogue(dial_left, l_text, _is_parenthetical(l_text) if l_text else False, line_num)
                append_to_dialogue(dial_right, r_text, _is_parenthetical(r_text) if r_text else False, line_num)
            else:
                append_to_dialogue(dial_left, text, primary_label == Label.P, line_num)
            continue

        if primary_label == Label.E:
            if dial_left is not None and text:
                dial_left.extensions.append(text)
            continue

    flush_narrative()
    flush_dialogues()
    return result


# ── Parser class ─────────────────────────────────────────────────────────────


class ScreenplayParser:
    def __init__(self, filepath: Path | str):
        self.filepath = Path(filepath)
        self.encoding = detect_encoding(self.filepath)
        self.meta: Metadata = None
        self.prelude: list[PreludeContent] = []
        self.scenes: list[Scene] = []

    @property
    def data(self) -> dict:
        return {
            "meta": [m.to_dict() for m in self.meta],
            "prelude": [p.to_dict() for p in self.prelude],
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def parse(self) -> None:
        self.meta: Metadata = None
        self.prelude: list[PreludeContent] = []
        self.scenes: list[Scene] = []

        raw_lines = self.filepath.read_text(encoding=self.encoding).splitlines()
        parsed_lines = self._parse_lines(raw_lines, LABELS_TO_IGNORE)

        id_generator = Counter()

        # ── Metadata content (before first non M line) ─────────────────────
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
                line_start_idx=metadata_buffer[0].line_idx,
                line_stop_idx=metadata_buffer[-1].line_idx,
            )

        # ── Prelude content (after last M line and before first S or D line) ─────────────────────
        first_scene_line_idx = self._find_first_occurrence(parsed_lines[first_non_meta_line_idx:], [Label.S, Label.D])
        previous_label: Label = None
        buffer: list[ParsedLine] = []

        def flush_buffer() -> None:
            nonlocal narrative_buffer
            if narrative_buffer:
                self.prelude.append(
                    Narrative(
                        id=id_generator.next(),
                        value=" ".join(t for t, _ in narrative_buffer),
                        line_start=narrative_buffer[0][1],
                        line_stop=narrative_buffer[-1][1],
                    )
                )
                narrative_buffer = []

        for pl in parsed_lines[first_non_meta_line_idx:first_scene_line_idx]:
            primary_label = pl.get_primary_label(LABELS_PRIORITY)
            if primary_label == Label.O:
                if previous_label is not None and previous_label not in LABELS_PRESERVING_CONTINUITY_AFTER_O:
                    flush_buffer()
                continue
            if not pl.sanitized_content:
                raise SCREENPLAYParsingError(f"Empty line not labeled `O` at {self.filepath}:{pl.line_idx}")
            if primary_label != previous_label:
                flush_buffer()
                previous_label = primary_label
            buffer.append(pl)

        flush_buffer()

        # ── Scenes ───────────────────────────────────────────────────────────
        # Scene boundaries: S lines (with S>S / S>O>S heading extension) and D
        # lines (which always start a fresh deleted scene, never merging).
        # O lines between two S lines are discarded; O lines between an S and
        # any other label are flushed into the current scene content as normal.
        current_heading: str | None = None
        current_deleted: bool = False
        current_heading_start: int = 0
        current_heading_stop: int = 0
        current_lines: list[ParsedLine] = []
        last_non_o_label: Label | None = None
        pending_o: list[ParsedLine] = []

        def flush_scene() -> None:
            built = _build_content(current_lines, ids)
            line_stop = built[-1].line_stop if built else current_heading_stop
            self.scenes.append(
                Scene(
                    id=ids.next(),
                    heading=current_heading,
                    content=built,
                    deleted=current_deleted,
                    line_start=current_heading_start,
                    line_stop=line_stop,
                )
            )

        for labels, content, line_num in parsed[first_scene:]:
            primary_label = extract_primary_label(labels, LABELS_PRIORITY)
            is_scene_boundary = Label.S in labels or primary_label == Label.D
            is_deletion = primary_label == Label.D and Label.S not in labels

            if is_scene_boundary:
                if not is_deletion and current_heading is not None and last_non_o_label == Label.S:
                    # S>S or S>O>S: extend the heading; discard buffered O lines.
                    current_heading += " " + content.strip()
                    current_heading_stop = line_num
                    pending_o = []
                else:
                    # Genuine new scene (new S not continuing a heading, or a D).
                    current_lines.extend(pending_o)
                    pending_o = []
                    if current_heading is not None:
                        flush_scene()
                    current_heading = content.strip()
                    current_deleted = is_deletion
                    current_heading_start = line_num
                    current_heading_stop = line_num
                    current_lines = []
                last_non_o_label = Label.D if is_deletion else Label.S

            elif primary_label == Label.O and last_non_o_label == Label.S:
                # O immediately after S: buffer it — might be the gap in S>O>S.
                pending_o.append((labels, content, line_num))

            else:
                # Any other line: commit buffered O lines and accumulate normally.
                current_lines.extend(pending_o)
                pending_o = []
                current_lines.append((labels, content, line_num))
                if primary_label != Label.O:
                    last_non_o_label = primary_label

        # End of file: flush any remaining O lines and the last open scene.
        current_lines.extend(pending_o)
        if current_heading is not None:
            flush_scene()

    @classmethod
    def read(cls, filepath: Path | str) -> Self:
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
        # TODO: Try to make this a staticmethod while preserving context in error
        parsed_lines = []
        for line_idx, raw in enumerate(raw_lines, start=1):
            if "|" not in raw:
                raise SCREENPLAYParsingError(f"No label found at {self.filepath}:{line_idx}")
            labels_str, content = raw.split("|", 1)
            parsed_lines.append((sanitize_labels(labels_str, labels_to_ignore), content, line_idx))
        return parsed_lines

    def _find_first_occurrence(self, parsed_lines: list[ParsedLine], labels: list[Label]) -> int:
        # TODO: Try to make this a staticmethod
        for pl_labels, _, line_idx in parsed_lines:
            if any(label in pl_labels for label in labels):
                return line_idx - 1
        raise SCREENPLAYParsingError(f"No label(s) {str(labels)} detected in {self.filepath}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.screenplay> [output.json]", file=sys.stderr)
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    parser = ScreenplayParser.read(src)
    output = parser.save(dst)

    scenes = len(parser.scenes)
    deleted = sum(1 for s in parser.scenes if s.deleted)
    items = sum(len(s.content) for s in parser.scenes)
    print(f"Parsed {scenes} scenes ({deleted} deleted), {items} content items -> {output}")


if __name__ == "__main__":
    main()
