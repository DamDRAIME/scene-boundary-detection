from pathlib import Path

from sbd.sprite.extractor.base import SpriteExtractor
from sbd.sprite.extractor.filehandler.mhtml import MHTMLFileHandler


class MHTMLSpriteExtractor(SpriteExtractor):
    def __init__(self, filehandler: MHTMLFileHandler):
        super().__init__(filehandler)

    @property
    def height(self) -> int:
        return self.filehandler.src_meta.sprite_shape[0]

    @property
    def width(self) -> int:
        return self.filehandler.src_meta.sprite_shape[1]

    @property
    def fps(self) -> float:
        return self.filehandler.src_meta.fps

    @property
    def mode(self) -> str:
        return "RGB"

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> "MHTMLSpriteExtractor":
        return cls(MHTMLFileHandler(filepath, **kwargs))
