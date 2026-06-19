from copy import deepcopy
import warnings

from sbd.subtitle.typings import SubTitle, SRTUtterance
from sbd.utils.counter import Counter


def subtitles_to_utterances(
    subtitles: list[SubTitle], end_sentence_markers: tuple[str] = (".", "!", "?", "-")
) -> list[SRTUtterance]:
    def flush_buffer():
        nonlocal buffer, utterances, id_generator
        utterances.append(SRTUtterance.from_subtitles(*buffer, idx=id_generator.next()))
        buffer = []

    def break_two_people_dialogue(subtitle: SubTitle):
        nonlocal buffer, utterances, id_generator
        content = subtitle.content.removeprefix("- ")
        parts = content.split(" - ")
        if len(parts) != 2:
            warnings.warn(
                f"Expected to find two parts for dialogue in Subtitle #{subtitle.idx} "
                f"at {subtitle.filepath}:{subtitle.line_idx}\n"
                f"Found {len(parts)}. It will be treated as one utterance."
            )
            utterances.append(SRTUtterance.from_subtitles(subtitle, idx=id_generator.next()))
            return
        first_u, second_u = parts
        utterances.append(
            SRTUtterance(
                idx=id_generator.next(),
                timestamp=subtitle.timestamp.first_half(),
                content=first_u,
                subtitles_indices=[subtitle.idx],
            )
        )
        if second_u.endswith(end_sentence_markers):
            utterances.append(
                SRTUtterance(
                    idx=id_generator.next(),
                    timestamp=subtitle.timestamp.second_half(),
                    content=second_u,
                    subtitles_indices=[subtitle.idx],
                )
            )
        else:
            subt = deepcopy(subtitle)
            subt.content = second_u
            subt.timestamp = subtitle.timestamp.second_half()
            buffer.append(subt)

    id_generator = Counter()
    utterances: list[SRTUtterance] = []
    buffer: list[SubTitle] = []
    for subtitle in subtitles:
        if subtitle.content.startswith("- "):
            if buffer:
                flush_buffer()
            break_two_people_dialogue(subtitle)
        elif subtitle.content.endswith(end_sentence_markers):
            buffer.append(subtitle)
            flush_buffer()
        else:
            buffer.append(subtitle)
    if buffer:
        flush_buffer()
    return utterances
