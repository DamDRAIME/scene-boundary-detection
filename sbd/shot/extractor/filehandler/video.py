from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import ffmpeg
import numpy as np

from sbd.exceptions import ShotExtractionError, VideoParsingError
from sbd.shot.extractor.filehandler.base import ShotFileHandler
from sbd.shot.extractor.filehandler.models import FrameImg, Shot, SourceMetadata
from sbd.sprite.extractor.filehandler.utils import resolve_shape


class VideoFileHandler(ShotFileHandler):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._src_meta = self.get_source_metadata(self.filepath)

    def extract_frame_at(
        self, timestamp: float, width: int = None, height: int = None, scale_ratio: float = None
    ) -> FrameImg:
        """Extract a single frame at the given timestamp.

        Args:
            timestamp (float): Timestamp in seconds.
            width (int, optional): Width to which the frame has to be resized. If None and a `height` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            height (int, optional): Height to which the frame has to be resized. If None and a `width` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            scale_ratio (float, optional): Rescaling to apply to each extracted frame. Mutually exclusive with
                `height` or `width`. Defaults to None.

        Returns:
            FrameImg: The extracted frame at the given timestamp.
        """
        height, width = self._resolve_shape(height, width, scale_ratio)
        expected_frame_size = width * height * 3
        output_kwargs = {"format": "rawvideo", "pix_fmt": "rgb24", "vframes": 1}
        run_kwargs = {"capture_stdout": True, "capture_stderr": True}
        stream = ffmpeg.input(self.filepath, ss=timestamp)
        if not self._is_source_shape(height, width):
            stream = stream.filter("scale", width, height)
        try:
            out, _ = stream.output("pipe:", **output_kwargs).run(**run_kwargs)
            if len(out) < expected_frame_size:
                raise ShotExtractionError(
                    f"Frame does not contain the expected number of pixels at timestamp {timestamp}."
                )
            frame = np.frombuffer(out, np.uint8).reshape((height, width, 3))
            return frame
        except Exception as e:
            raise ShotExtractionError(
                f"An error occurred during the extraction of the frame at timestamp {timestamp}."
            ) from e

    @staticmethod
    def get_source_metadata(filepath: str | Path) -> SourceMetadata:
        try:
            meta = ffmpeg.probe(filepath)
        except Exception as e:
            raise VideoParsingError("Invalid file type") from e
        video_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
        if not video_stream:
            raise VideoParsingError("Could not find a stream with a `video` codec type.")
        fps = float(Fraction(video_stream["r_frame_rate"]))
        duration = float(video_stream.get("duration") or meta["format"]["duration"])
        return SourceMetadata(
            fps=fps,
            duration=duration,
            frame_shape=(int(video_stream["height"]), int(video_stream["width"])),
        )

    def _resolve_shape(self, height: int = None, width: int = None, scale_ratio: float = None) -> tuple[int, int]:
        if scale_ratio and (height or width):
            raise ValueError("Pass either `scale_ratio` or `height`/`width`, not both.")
        return resolve_shape(self.src_meta.frame_shape, (height, width), scale_ratio)

    def _is_source_shape(self, height: int, width: int) -> bool:
        return self.src_meta.frame_shape == (height, width)
