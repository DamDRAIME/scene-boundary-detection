from pathlib import Path
from typing import Self

from sentence_transformers import SentenceTransformer

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

    def get_singleton_utterances(self, embedding_model: SentenceTransformer | None = None) -> list[SingletonUtterance]:
        utterances = screenplay_scenes_to_singleton_utterances(self.scenes)
        if embedding_model is None:
            return utterances
        # Generate embeddings for each utterance
        for utterance in utterances:
            utterance.embedding = embedding_model.encode(utterance.content).tolist()
        return utterances
