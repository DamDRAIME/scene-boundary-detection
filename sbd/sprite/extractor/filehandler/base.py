from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from sbd.sprite.extractor.filehandler.models import SourceMetadata, SpriteImg


class FileHandler(ABC):
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Input file not found: {str(self.filepath)}")

    @property
    def src_meta(self) -> SourceMetadata:
        if not self._src_meta:
            self._src_meta = self.get_source_metadata()
        return self._src_meta

    @abstractmethod
    def iter_sprites(self, *args, **kwargs) -> Iterator[tuple[float, SpriteImg]]:
        pass

    @abstractmethod
    def get_source_metadata(self) -> SourceMetadata:
        pass
