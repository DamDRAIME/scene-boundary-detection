from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import ffmpeg

from sbd.sprite.extractor.exceptions import SpriteExtractionError, VideoParsingError
from sbd.sprite.extractor.base import FileHandler
from sbd.sprite.extractor.filehandler.models import ExtractionMethod, SourceMetadata, SpriteImg
from sbd.sprite.extractor.filehandler.utils import resolve_shape


class VideoFileHandler(FileHandler):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._src_meta = self.get_source_metadata()

    def iter_sprites(
        self,
        fps: float = 1.0,
        height: int = None,
        width: int = None,
        scale_ratio: float = None,
        method: ExtractionMethod = ExtractionMethod.SELECT,
    ) -> Iterator[tuple[float, SpriteImg]]:
        """Yield `(timestamp, sprite)` pairs sampled from the video at `fps`, using given `method`.

        Args:
            fps (float, optional): Number of frames/sprites to extract every second. Defaults to 1.0.
            height (int, optional): Height to which the frame has to be resized. If None and a `width` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            width (int, optional): Width to which the frame has to be resized. If None and a `height` is set, it will
                be inferred so as to keep the original height/width ratio. Mutually exclusive with `scale_ratio`.
                Defaults to None.
            scale_ratio (float, optional): Rescaling to apply to each extracted frame. Mutually exclusive with
                `height` or `width`. Defaults to None.
            method (ExtractionMethod, optional): Approach to use for the extraction. See `extract()` for more info.
                Defaults to SELECT.

        Yields:
            Iterator[tuple[float, np.ndarray]]: Sampled `(timestamp, sprite)` pairs.
        """
        height, width = self._resolve_shape(height, width, scale_ratio)
        method = ExtractionMethod(method)

        if method is ExtractionMethod.SELECT:
            return self._iter_frames_select(fps, width, height)
        return self._iter_frames_seek(fps, width, height)

    def get_source_metadata(self) -> SourceMetadata:
        meta = self.get_source_raw_metadata(self.filepath)
        video_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
        if not video_stream:
            raise VideoParsingError("Could not find a stream with a `video` codec type.")
        return SourceMetadata(
            fps=float(Fraction(video_stream["r_frame_rate"])),
            duration=float(video_stream.get("duration") or meta["format"]["duration"]),
            n_sprites=int(video_stream["nb_frames"]),
            sprite_shape=(int(video_stream["height"]), int(video_stream["width"])),
        )

    @staticmethod
    def get_source_raw_metadata(filepath: str | Path) -> dict[str, Any]:
        try:
            return ffmpeg.probe(filepath)
        except Exception as e:
            raise VideoParsingError("Invalid file type") from e

    def _iter_frames_select(self, fps: float, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
        frame_size = width * height * 3
        n = max(1, round(self.src_meta.fps / fps))
        stream = ffmpeg.input(self.filepath).filter("select", f"not(mod(n,{n}))")
        if not self._is_source_shape(height, width):
            stream = stream.filter("scale", width, height)

        process = (
            stream.output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", fps_mode="vfr"
            )  # vfr: otherwise ffmpeg re-duplicates dropped frames to match the source's CFR.
            .global_args("-loglevel", "error")  # Keep stderr near-silent: a chatty stderr can fill its OS pipe
            .run_async(pipe_stdout=True, pipe_stderr=True)  # buffer and deadlock the whole pipeline on long videos.
        )
        try:
            frame_idx = 0
            while True:
                in_bytes = process.stdout.read(frame_size)
                if len(in_bytes) < frame_size:
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
                timestamp = frame_idx / fps
                yield timestamp, frame
                frame_idx += 1
        except Exception as e:
            raise SpriteExtractionError("An error occurred during the extraction of the sprite.") from e
        finally:
            process.stdout.close()
            stderr = process.stderr.read()
            process.stderr.close()
            returncode = process.wait()
            if returncode != 0:
                raise SpriteExtractionError(
                    "An error occurred during the extraction of the sprite: "
                    f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')}"
                )

    def _iter_frames_seek(self, fps: float, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
        output_kwargs = {"format": "rawvideo", "pix_fmt": "rgb24", "vframes": 1}
        run_kwargs = {"capture_stdout": True, "capture_stderr": True}
        is_source_shape = self._is_source_shape(height, width)
        frame_size = width * height * 3
        n_frames = int(self.src_meta.duration * fps) + 1
        for frame_idx in range(n_frames):
            try:
                timestamp = frame_idx / fps
                stream = ffmpeg.input(self.filepath, ss=timestamp)
                if not is_source_shape:
                    stream = stream.filter("scale", width, height)
                out, _ = stream.output("pipe:", **output_kwargs).run(**run_kwargs)
                if len(out) < frame_size:
                    break
                frame = np.frombuffer(out, np.uint8).reshape((height, width, 3))
                yield timestamp, frame
            except Exception as e:
                raise SpriteExtractionError("An error occurred during the extraction of the sprite.") from e

    def _resolve_shape(self, height: int = None, width: int = None, scale_ratio: float = None) -> tuple[int, int]:
        if scale_ratio and (height or width):
            raise ValueError("Pass either `scale_ratio` or `height`/`width`, not both.")
        return resolve_shape(self.src_meta.sprite_shape, (height, width), scale_ratio)

    def _is_source_shape(self, height: int, width: int) -> bool:
        return self.src_meta.sprite_shape == (height, width)
