import re
from itertools import groupby

from sbd.common.utils.counter import Counter
from sbd.screenplay.models import Dialogue, Label, ParsedLine, Scene, ScreenplayChildElement, ScreenplayElement


def sanitize_labels(labels_raw: str, labels_to_ignore: frozenset[Label]) -> frozenset[Label]:
    labels = {Label[label.strip()] for label in labels_raw.split(",")}
    reduced = labels - labels_to_ignore
    return frozenset(reduced) if reduced else frozenset(labels)


def clean_parenthetical(text: str) -> str:
    return text.strip().removeprefix("(").removesuffix(")").strip()


def split_character_and_extensions(text: str) -> tuple[str, list[str]]:
    exts = re.findall(r"\(([^)]+)\)", text)
    name = re.sub(r"\s*\([^)]+\)", "", text).strip().rstrip(".")
    return name, exts


def build_scene(content: list[ScreenplayElement], id: Counter, heading: list[ParsedLine], act: str | None) -> Scene:
    return Scene(
        id=id.next(),
        source_line_start_idx=content[0].source_line_start_idx,
        source_line_stop_idx=content[-1].source_line_stop_idx,
        recently_modified=any(el.recently_modified for el in content),
        heading=" ".join([s.sanitized_content for s in heading]),
        act=act,
        deleted=heading[0].primary_label == Label.D if heading else False,
        content=content,
    )


def build_screenplay_element(buffer: list[ParsedLine], id: Counter) -> ScreenplayElement:
    if buffer[0].primary_label == Label.C:
        return build_dialogue_element(buffer, id)
    return build_child_element(buffer, id)


def build_dialogue_element(buffer: list[ParsedLine], id: Counter) -> Dialogue:
    dialogue_id = id.next()
    character_line = buffer[0]
    character_name, extensions = split_character_and_extensions(character_line.sanitized_content)
    content = []
    for label, parsed_lines in groupby(buffer[1:], lambda x: x.primary_label):
        if label == Label.E:
            extensions.extend([pl.sanitized_content for pl in parsed_lines])
            continue
        content.append(build_child_element(list(parsed_lines), id))
    return Dialogue(
        id=dialogue_id,
        source_line_start_idx=buffer[0].line_idx,
        source_line_stop_idx=buffer[-1].line_idx,
        recently_modified=any(b.is_stared for b in buffer),
        content=content,
        character=character_name,
        extensions=extensions,
    )


def build_child_element(buffer: list[ParsedLine], id: Counter) -> ScreenplayChildElement:
    primary_label = buffer[0].get_primary_label()
    expected_labels = (Label.N, Label.T, Label.M, Label.Y, Label.P, Label.U)
    if primary_label not in expected_labels:
        from pprint import pprint

        pprint(buffer)
        raise NotImplementedError(
            f"Unsupported Screenplay Element. Expected one of {expected_labels}, got {primary_label.value}"
        )
    value = [pl.sanitized_content for pl in buffer]
    if primary_label == Label.P:
        value = [clean_parenthetical(v) for v in value]
    return ScreenplayChildElement(
        id=id.next(),
        source_line_start_idx=buffer[0].line_idx,
        source_line_stop_idx=buffer[-1].line_idx,
        recently_modified=any(pl.is_stared for pl in buffer),
        value=value,
        _type=primary_label,
    )


# ── Character-line parsing ───────────────────────────────────────────────────


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
        return [split_character_and_extensions(_span(groups[0]))]

    gaps = [(groups[i + 1][0].start() - groups[i][-1].end(), i) for i in range(len(groups) - 1)]
    max_gap, max_idx = max(gaps)

    if max_gap <= 6:
        return [split_character_and_extensions(stripped[groups[0][0].start() : groups[-1][-1].end()])]

    left_g, right_g = groups[: max_idx + 1], groups[max_idx + 1 :]
    return [
        split_character_and_extensions(stripped[left_g[0][0].start() : left_g[-1][-1].end()]),
        split_character_and_extensions(stripped[right_g[0][0].start() : right_g[-1][-1].end()]),
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
