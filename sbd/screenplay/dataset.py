from pathlib import Path
from typing import Self

from sbd.screenplay.models import Metadata, Scene, SingletonUtterance
from sbd.screenplay.parser import ScreenplayParser
from sbd.screenplay.utterance import screenplay_scenes_to_singleton_utterances


class ScreenplayDataset:
    def __init__(self, filepath: Path | str, metadata: Metadata, scenes: list[Scene]):
        self.filepath = Path(filepath)
        self.metadata = metadata
        self.scenes = scenes

    @classmethod
    def read(cls, filepath: Path | str) -> Self:
        metadata, scenes = ScreenplayParser(filepath)()
        return cls(filepath, metadata, scenes)

    def get_singleton_utterances(self) -> list[SingletonUtterance]:
        return screenplay_scenes_to_singleton_utterances(self.scenes)
