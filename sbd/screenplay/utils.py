from itertools import groupby

from sbd.screenplay.models import (
    Chyron,
    Dialogue,
    Label,
    Metadata,
    Narrative,
    ParsedLine,
    ScreenplayChildElement,
    ScreenplayElement,
    Transition,
)
from sbd.shared.utils.counter import Counter


def sanitize_labels(labels_raw: str, labels_to_ignore: frozenset[Label]) -> frozenset[Label]:
    labels = {Label[label.strip()] for label in labels_raw.split(",")}
    reduced = labels - labels_to_ignore
    return frozenset(reduced) if reduced else frozenset(labels)


def build_screenplay_element(buffer: list[ParsedLine], id: Counter) -> ScreenplayElement:
    if buffer[0].primary_label == Label.C:
        return build_dialogue_element(buffer, id)
    return build_child_element(buffer, id)


def build_dialogue_element(buffer: list[ParsedLine], id: Counter) -> Dialogue:
    dialogue_id = id.next()
    character_line = buffer[0]  # TODO: Extract extension(s) out of this line
    extensions = []
    content = []
    for label, parsed_lines in groupby(buffer[1:], lambda x: x.primary_label):
        if label == Label.E:
            extensions.extend([pl.sanitized_content for pl in parsed_lines])
            continue
        content.append(build_child_element(parsed_lines, id))
    return Dialogue(
        id=dialogue_id,
        line_start_idx=buffer[0].line_idx,
        line_stop_idx=buffer[-1].line_idx,
        recently_modified=any(b.is_stared for b in buffer),
        content=content,
        character=character_line.sanitized_content,
        extensions=extensions,
    )


def build_child_element(buffer: list[ParsedLine], id: Counter) -> ScreenplayChildElement:
    primary_label = buffer[0].get_primary_label()
    expected_labels = (Label.N, Label.T, Label.M, Label.Y)
    if primary_label not in expected_labels:
        raise NotImplementedError(
            f"Unsupported Screenplay Element. Expected one of {expected_labels}, got {primary_label.value}"
        )
    return ScreenplayChildElement(
        id=id.next(),
        line_start_idx=buffer[0].line_idx,
        line_stop_idx=buffer[-1].line_idx,
        recently_modified_lines_idx=[pl.line_idx for pl in buffer if pl.is_stared],
        value=[pl.sanitized_content for pl in buffer],
        _type=primary_label,
    )
