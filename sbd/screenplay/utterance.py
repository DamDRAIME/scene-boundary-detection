from sbd.common.utils.counter import Counter
from sbd.screenplay.models import Dialogue, Label, Scene, SingletonUtterance
from sbd.screenplay.utils import split_into_sentences


def screenplay_scenes_to_singleton_utterances(scenes: list[Scene]) -> list[SingletonUtterance]:
    id_generator = Counter()
    utterances: list[SingletonUtterance] = []
    for scene in scenes:
        for element in scene.content:
            if not isinstance(element, Dialogue):
                continue
            for child in element.content:
                if child._type != Label.U:
                    continue
                content = " ".join(child.value) if isinstance(child.value, list) else child.value
                for sentence in split_into_sentences(content):
                    utterances.append(
                        SingletonUtterance(
                            id=id_generator.next(),
                            scene_id=scene.id,
                            scene_heading=scene.heading,
                            character=element.character,
                            content=sentence,
                            source_line_start_idx=child.source_line_start_idx,
                            source_line_stop_idx=child.source_line_stop_idx,
                        )
                    )
    return utterances
