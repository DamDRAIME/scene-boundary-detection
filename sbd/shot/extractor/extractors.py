from datetime import timedelta
from pathlib import Path
from typing import Iterator, Self

import ffmpeg

from sbd.shot.extractor.base import ShotExtractor
from sbd.shot.extractor.filehandler.models import Shot
from sbd.shot.extractor.filehandler.video import VideoFileHandler


class FFMPEGShotExtractor(ShotExtractor):
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
        threshold: float = 0.2,
        height: int = None,
        width: int = None,
        scale_ratio: float = None,
    ) -> Path:
        """Extract a frame at each shot boundary, optionally rescale it, and save all frames, along with their
        respective timestamp, and score, to an HDF5 file.

        Writes three datasets to `output_filepath`:
            - `sprites`
                - Shape: `(n_shots, height, width, 3)` (uint8, RGB)
                - Attributes: `source`, `n_entries`, `height`, and `width`
            - `timestamps`
                - Shape: `(n_shots,)` (float64, seconds)
                - Attributes: `unit`
            - `scores`
                - Shape: `(n_shots,)` (float64)
        where `timestamps[i]` is the timestamp of `sprites[i]`, and `scores[i]` its score given by FFMPEG's shot
        boundary detection algorithm.

        Args:
            output_filepath (str | Path): Filepath of the resulting HDF5 datasets (forced to .h5).
            threshold (float, optional): Threshold for shot boundary detection. Defaults to 0.2.
            height (int, optional): Height to which the frame has to be resized. If None and a `width` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            width (int, optional): Width to which the frame has to be resized. If None and a `height` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            scale_ratio (float, optional): Rescaling to apply to each extracted frame. Mutually exclusive with
                `height` or `width`. Defaults to None.

        Returns:
            Path: Filepath of the resulting HDF5 file.
        """
        self._height, self._width = self.filehandler._resolve_shape(height, width, scale_ratio)
        return super().extract(output_filepath, threshold=threshold, height=self._height, width=self._width)

    def iter_shots(
        self, threshold: float = 0.2, height: int = None, width: int = None, scale_ratio: float = None
    ) -> Iterator[Shot]:
        if 0 >= threshold or threshold >= 1:
            raise ValueError("Threshold must be greater than 0 and less than 1.")
        probe_result = ffmpeg.probe(
            "movie=" + self.filehandler.filepath.as_posix() + r",select=gt(scene\," + str(threshold) + ")",
            f="lavfi",
            show_frames=None,
        )
        for shot_data in probe_result["frames"]:
            ts = float(shot_data["pts_time"])
            frame = self.filehandler.extract_frame_at(
                timestamp=ts,
                width=width,
                height=height,
                scale_ratio=scale_ratio,
            )
            yield Shot(
                frame=frame,
                timestamp=timedelta(seconds=ts),
                score=float(shot_data["tags"]["lavfi.scene_score"]),
            )

    @classmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        return cls(VideoFileHandler(filepath, **kwargs))
