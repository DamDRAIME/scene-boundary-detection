import tempfile
from pathlib import Path
from typing import Any, Iterator

import ffmpeg

from sbd.exceptions import VideoParsingError
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.extractor.filehandler.srt import SRTFileHandler
from sbd.subtitle.models import SubTitle, SubtitleStreamMetadata


class VideoFileHandler(SubtitleFileHandler):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._src_meta = self.get_source_metadata()

    def iter_subtitles(self, stream_idx: int = None) -> Iterator[SubTitle]:
        stream_meta = self.get_subtitle_stream(stream_idx)
        if stream_meta.codec_name == "subrip":
            yield from self._iter_subrip_subtitles(stream_meta)
        else:
            raise VideoParsingError(
                f"Unsupported subtitle codec '{stream_meta.codec_name}' in stream index {stream_meta.index} of {self.filepath}."
            )

    def _iter_subrip_subtitles(self, stream_meta: SubtitleStreamMetadata) -> Iterator[SubTitle]:
        # `delete=False` + closing the handle before invoking ffmpeg is required on Windows,
        # where an open NamedTemporaryFile holds an exclusive lock that blocks ffmpeg from writing to it.
        named_file = tempfile.NamedTemporaryFile(mode="w+t", suffix=".srt", delete=False)
        named_file.close()
        stream = ffmpeg.input(str(self.filepath))
        try:
            stream.output(named_file.name, map=f"0:{stream_meta.index}", **{"c:s": "srt"}).run(
                overwrite_output=True, quiet=True
            )

            # Read the temporary file and yield SubTitle objects
            srt_handler = SRTFileHandler(named_file.name)
            for subtitle in srt_handler.iter_subtitles():
                subtitle.filepath = Path(self.filepath)  # Set the original video file path not the temporary file path
                yield subtitle
        finally:
            Path(named_file.name).unlink()

    def get_source_metadata(self) -> list[SubtitleStreamMetadata]:
        meta = self.get_source_raw_metadata(self.filepath)
        subtitle_streams = [s for s in meta["streams"] if s["codec_type"] == "subtitle"]
        if not subtitle_streams:
            raise VideoParsingError("Could not find a stream with a `subtitle` codec type.")
        return [SubtitleStreamMetadata.from_ffprobe(s) for s in subtitle_streams]

    @staticmethod
    def get_source_raw_metadata(filepath: str | Path) -> dict[str, Any]:
        try:
            return ffmpeg.probe(str(filepath))
        except Exception as e:
            raise VideoParsingError("Invalid file type") from e

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
