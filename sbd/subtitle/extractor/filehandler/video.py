import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import ffmpeg

from sbd.exceptions import VideoParsingError
from sbd.subtitle.extractor.filehandler.ass import ASSFileHandler
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.extractor.filehandler.srt import SRTFileHandler
from sbd.subtitle.models import SubTitle, SubtitleStreamMetadata

# Maps a subtitle codec name (as reported by ffprobe) to the file suffix/codec to pass to
# ffmpeg and the file handler used to parse the resulting temporary file.
CODEC_HANDLERS: dict[str, type[SubtitleFileHandler]] = {
    "subrip": SRTFileHandler,
    "ass": ASSFileHandler,
}


class VideoFileHandler(SubtitleFileHandler):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._src_meta = self.get_source_metadata(self.filepath)

    def iter_subtitles(self, stream_idx: int = None) -> Iterator[SubTitle]:
        stream_meta = self.get_subtitle_stream(stream_idx)
        codec_handler = CODEC_HANDLERS.get(stream_meta.codec_name)
        if codec_handler is None:
            raise VideoParsingError(
                f"Unsupported subtitle codec '{stream_meta.codec_name}' in stream index {stream_meta.index} of "
                f"{self.filepath}. Currently supported codecs: {', '.join(CODEC_HANDLERS.keys())}"
            )
        with self._extract_stream_to_temp_file(stream_meta, codec_handler.file_suffix) as temp_filepath:
            handler = codec_handler(temp_filepath)
            for subtitle in handler.iter_subtitles():
                subtitle.filepath = Path(self.filepath)  # Set the original video file path not the temporary file path
                yield subtitle

    @contextmanager
    def _extract_stream_to_temp_file(self, stream_meta: SubtitleStreamMetadata, suffix: str) -> Iterator[Path]:
        # `delete=False` + closing the handle before invoking ffmpeg is required on Windows,
        # where an open NamedTemporaryFile holds an exclusive lock that blocks ffmpeg from writing to it.
        named_file = tempfile.NamedTemporaryFile(mode="w+t", suffix=suffix, delete=False)
        named_file.close()
        try:
            ffmpeg.input(str(self.filepath)).output(
                named_file.name, map=f"0:{stream_meta.index}", **{"c:s": suffix.lstrip(".")}
            ).run(overwrite_output=True, quiet=True)
            yield Path(named_file.name)
        finally:
            Path(named_file.name).unlink()

    @staticmethod
    def get_source_metadata(filepath: str | Path) -> list[SubtitleStreamMetadata]:
        try:
            meta = ffmpeg.probe(str(filepath))
        except Exception as e:
            raise VideoParsingError("Invalid file type") from e
        subtitle_streams = [s for s in meta["streams"] if s["codec_type"] == "subtitle"]
        if not subtitle_streams:
            raise VideoParsingError("Could not find a stream with a `subtitle` codec type.")
        return [SubtitleStreamMetadata.from_ffprobe(s) for s in subtitle_streams]

    def get_subtitle_stream(self, stream_idx: int | None) -> SubtitleStreamMetadata:
        if stream_idx is None:
            if len(self._src_meta) == 1:
                return self._src_meta[0]
            raise VideoParsingError(
                f"Multiple subtitle streams found in {self.filepath}. Please specify a stream index between 0 and {len(self._src_meta) - 1}."
            )
        for s in self._src_meta:
            if s.index == stream_idx:
                return s
        raise VideoParsingError(f"Subtitle stream index {stream_idx} not found in {self.filepath}.")
