from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from sbd.shot.extractor.filehandler.models import FrameImg, SourceMetadata


class ShotFileHandler(ABC):
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
    def extract_frame_at(self, timestamp: float) -> FrameImg:
        pass

    @abstractmethod
    def get_source_metadata(self) -> SourceMetadata:
        pass
