from pathlib import Path
from typing import Iterator, Self

from sbd.subtitle.extractor.base import SubtitleExtractor
from sbd.subtitle.extractor.filehandler.models import SubTitle, Utterance
from sbd.subtitle.extractor.filehandler.srt import SRTFileHandler
from sbd.subtitle.extractor.utterance import subtitles_to_utterances


class SRTSubtitleExtractor(SubtitleExtractor):
    def __init__(self, filehandler: SRTFileHandler, as_utterances: bool = False):
        super().__init__(filehandler, data_type="subtitle" if not as_utterances else "utterance")
        self.as_utterances = as_utterances

    @classmethod
    def from_file(cls, filepath: str | Path, as_utterances: bool = False, **kwargs) -> Self:
        return cls(SRTFileHandler(filepath, **kwargs), as_utterances=as_utterances)

    def iter_subtitles(self, *args, **kwargs) -> Iterator[SubTitle | Utterance]:
        if self.as_utterances:
            yield from self.iter_utterances(*args, **kwargs)
        else:
            yield from self.filehandler.iter_subtitles(*args, **kwargs)

    def iter_utterances(self, *args, **kwargs) -> Iterator[Utterance]:
        subtitles = list(self.filehandler.iter_subtitles(*args, **kwargs))
        yield from subtitles_to_utterances(subtitles)
