from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from sbd.subtitle.models import SubTitle


class SubtitleFileHandler(ABC):
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Input file not found: {str(self.filepath)}")

    @abstractmethod
    def iter_subtitles(self, *args, **kwargs) -> Iterator[SubTitle]:
        pass
