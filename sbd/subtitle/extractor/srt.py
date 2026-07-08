from pathlib import Path
from typing import Self

from sbd.subtitle.extractor.base import SubtitleExtractor
from sbd.subtitle.extractor.filehandler.srt import SRTFileHandler


class SRTSubtitleExtractor(SubtitleExtractor):
    def __init__(self, filehandler: SRTFileHandler):
        super().__init__(filehandler, data_type="subtitle")

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(SRTFileHandler(filepath, **kwargs))
