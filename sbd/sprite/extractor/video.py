from pathlib import Path
from typing import Self

from sbd.sprite.extractor.base import SpriteExtractor
from sbd.sprite.extractor.filehandler.models import ExtractionMethod
from sbd.sprite.extractor.filehandler.video import VideoFileHandler


class VideoSpriteExtractor(SpriteExtractor):
    def __init__(self, filehandler: VideoFileHandler):
        super().__init__(filehandler)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def mode(self) -> str:
        return "RGB"

    def extract(
        self,
        output_filepath: str | Path,
        fps: float = 1.0,
        height: int = None,
        width: int = None,
        scale_ratio: float = None,
        method: ExtractionMethod = ExtractionMethod.SELECT,
    ) -> Path:
        """Extract one frame/sprite every 1/`fps` second(s), optionally rescale it, and save all frames, along with
        their respective timestamp, to an HDF5 file.

        Writes two datasets to `output_filepath`:
            - `sprites`
                - Shape: `(n_sprites, height, width, 3)` (uint8, RGB)
                - Attributes: `source`, `n_sprites`, `method`, `height`, `width`, and `fps`
            - `timestamps`
                - Shape: `(n_sprites,)` (float64, seconds)
                - Attributes: `unit`
        where `timestamps[i]` is the timestamp of `sprites[i]`.

        Args:
            output_filepath (str | Path): Filepath of the resulting HDF5 datasets (forced to .h5).
            fps (float, optional): Number of frames/sprites to extract every second. Defaults to 1.0.
            height (int, optional): Height to which the frame has to be resized. If None and a `width` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            width (int, optional): Width to which the frame has to be resized. If None and a `height` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            scale_ratio (float, optional): Rescaling to apply to each extracted frame. Mutually exclusive with
                `height` or `width`. Defaults to None.
            method (ExtractionMethod, optional): Approach to use for the extraction. 2 options:
                - `SELECT`: one ffmpeg process, decodes the whole video once through the `select=not(mod(n,N))` filter,
                    keeping every Nth frame. Accurate but its cost scales with video *length*, not with the number of
                    samples requested -- slow when sampling sparsely from a long video.
                - `SEEK`: one ffmpeg process per sample, accurate seek (`-ss` + decode forward from the nearest keyframe).
                    Its cost scales with the number of *samples*, not video length -- much faster for sparse sampling, but
                    occasionally lands a frame off.

                Defaults to SELECT.

        Returns:
            Path: Filepath of the resulting HDF5 file.
        """
        self._fps = fps
        self._height, self._width = self.filehandler._resolve_shape(height, width, scale_ratio)
        self.method = ExtractionMethod(method).value
        iter_sprites_kwargs = {"fps": self._fps, "height": self._height, "width": self._width, "method": self.method}
        return super().extract(output_filepath, **iter_sprites_kwargs)

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(VideoFileHandler(filepath, **kwargs))
