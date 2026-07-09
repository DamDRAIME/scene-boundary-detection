from pathlib import Path
from typing import Self

from sbd.subtitle.extractor.base import SubtitleExtractor
from sbd.subtitle.extractor.filehandler.ass import ASSFileHandler
from sbd.subtitle.extractor.filehandler.srt import SRTFileHandler
from sbd.subtitle.extractor.filehandler.video import VideoFileHandler


class ASSSubtitleExtractor(SubtitleExtractor):
    def __init__(self, filehandler: ASSFileHandler):
        super().__init__(filehandler)

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(ASSFileHandler(filepath, **kwargs))


class SRTSubtitleExtractor(SubtitleExtractor):
    def __init__(self, filehandler: SRTFileHandler):
        super().__init__(filehandler)

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(SRTFileHandler(filepath, **kwargs))


class VideoSubtitleExtractor(SubtitleExtractor):
    def __init__(self, filehandler: Self):
        super().__init__(filehandler)

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(VideoFileHandler(filepath, **kwargs))
