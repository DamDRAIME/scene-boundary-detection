from enum import auto, StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import h5py
import ffmpeg

from sbd.sprite.exceptions import VideoSpriteExtractionError


class ExtractionMethod(StrEnum):
    SELECT = auto()
    SEEK = auto()


ExtractionMethod("test")


class VideoSpriteExtractor:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Video file not found: {str(self.filepath)}")
        self.meta = self.get_metadata(self.filepath)
        video_stream = next(s for s in self.meta["streams"] if s["codec_type"] == "video")
        if not video_stream:
            raise VideoSpriteExtractionError("Could not find a stream with a `video` codec type.")
        self.duration = float(video_stream.get("duration") or self.meta["format"]["duration"])
        self.source_width = int(video_stream["width"])
        self.source_height = int(video_stream["height"])
        self.source_fps = float(Fraction(video_stream["r_frame_rate"]))

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
        output_filepath = Path(output_filepath).with_suffix(".h5")
        width, height = self._resolve_size(height, width, scale_ratio)

        with h5py.File(output_filepath, "w") as h5_fh:
            sprites = h5_fh.create_dataset(
                "sprites",
                shape=(0, height, width, 3),
                maxshape=(None, height, width, 3),
                dtype="uint8",
                chunks=(1, height, width, 3),  # Chunked by sprite for performance at retrieval
            )
            timestamps = h5_fh.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="float64")

            sprite_idx = -1
            for sprite_idx, (timestamp, frame) in enumerate(self.iter_frames(fps, height, width, method=method)):
                sprites.resize(sprite_idx + 1, axis=0)
                sprites[sprite_idx] = frame
                timestamps.resize(sprite_idx + 1, axis=0)
                timestamps[sprite_idx] = timestamp

            sprites.attrs["source"] = self.filepath.stem
            sprites.attrs["n_sprites"] = sprite_idx + 1
            sprites.attrs["method"] = ExtractionMethod(method).value
            for attr_name in ("height", "width", "fps"):
                sprites.attrs[attr_name] = locals()[attr_name]
            timestamps.attrs["unit"] = "seconds"

        return output_filepath

    def iter_frames(
        self,
        fps: float = 1.0,
        height: int = None,
        width: int = None,
        scale_ratio: float = None,
        method: ExtractionMethod = ExtractionMethod.SELECT,
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield `(timestamp, frame)` pairs sampled from the video at `fps`, using given `method`.

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

        Returns:
            _type_: _description_

        Yields:
            Iterator[tuple[float, np.ndarray]]: Sampled `(timestamp, frame)` pairs.
        """
        height, width = self._resolve_size(height, width, scale_ratio)
        method = ExtractionMethod(method)

        if method is ExtractionMethod.SELECT:
            return self._iter_frames_select(fps, width, height)
        return self._iter_frames_seek(fps, width, height)

    @staticmethod
    def get_metadata(filepath: str | Path) -> dict[str, Any]:
        return ffmpeg.probe(filepath)

    def _iter_frames_select(self, fps: float, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
        frame_size = width * height * 3
        n = max(1, round(self.source_fps / fps))
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
            raise VideoSpriteExtractionError("An error occurred during the extraction of the sprite.") from e
        finally:
            process.stdout.close()
            stderr = process.stderr.read()
            process.stderr.close()
            returncode = process.wait()
            if returncode != 0:
                raise VideoSpriteExtractionError(
                    "An error occurred during the extraction of the sprite: "
                    f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')}"
                )

    def _iter_frames_seek(self, fps: float, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
        output_kwargs = {"format": "rawvideo", "pix_fmt": "rgb24", "vframes": 1}
        run_kwargs = {"capture_stdout": True, "capture_stderr": True}
        is_source_shape = self._is_source_shape(height, width)
        frame_size = width * height * 3
        n_frames = int(self.duration * fps) + 1
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
                raise VideoSpriteExtractionError("An error occurred during the extraction of the sprite.") from e

    def _resolve_size(self, height: int = None, width: int = None, scale_ratio: float = None) -> tuple[int, int]:
        if scale_ratio and (height or width):
            raise ValueError("Pass either `scale_ratio` or `height`/`width`, not both.")
        if scale_ratio:
            return int(self.source_width * scale_ratio), int(self.source_height * scale_ratio)
        if height and not width:
            return int(self.source_width * height / self.source_height), height
        if width and not height:
            return width, int(self.source_height * width / self.source_width)
        if width and height:
            return width, height
        return self.source_width, self.source_height

    def _is_source_shape(self, height: int, width: int) -> bool:
        return width == self.source_width and height == self.source_height
