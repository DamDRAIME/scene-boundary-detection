from pathlib import Path
from typing import Optional

import typer
from InquirerPy import get_style, inquirer
from InquirerPy.base.control import Choice

from sbd.subtitle.extractor.extractors import ASSSubtitleExtractor, SRTSubtitleExtractor, VideoSubtitleExtractor
from sbd.subtitle.extractor.filehandler.video import VideoFileHandler

app = typer.Typer()

SUBTITLE_EXTRACTOR_BY_SUFFIX = {
    ".srt": SRTSubtitleExtractor,
    ".ass": ASSSubtitleExtractor,
}


@app.callback()
def callback():
    """Scene Boundary Detection (SBD) CLI tool for extracting subtitles from video files or subtitle files."""


STREAM_PICKER_STYLE = get_style(
    {
        "questionmark": "#61afef bold",
        "question": "bold",
        "answermark": "#98c379 bold",
        "answer": "#98c379 bold",
        "pointer": "#61afef bold",
        "instruction": "#5c6370 italic",
    },
    style_override=False,
)


def _select_stream_idx(filepath: Path) -> int:
    streams = VideoFileHandler(filepath).get_source_metadata()
    if len(streams) == 1:
        return streams[0].index

    codec_width = max(len(stream.codec_name) for stream in streams)
    lang_width = max(len(stream.language or "und") for stream in streams)
    choices = [
        Choice(
            value=stream.index,
            name=f"[#{stream.index:<3}] {stream.codec_name.upper():<{codec_width}}  "
            f"{(stream.language or 'und').upper():<{lang_width}}  {stream.title or '(untitled)'}",
        )
        for stream in streams
    ]
    return inquirer.select(
        message=f"Multiple subtitle streams found in {filepath.name}:",
        choices=choices,
        instruction="(Use arrow keys, Enter to confirm)",
        style=STREAM_PICKER_STYLE,
    ).execute()


@app.command()
def extract_subtitles(
    input_path: Path = typer.Argument(..., exists=True, help="Path to the subtitle or video file."),
    output_path: Path = typer.Argument(..., help="Path to write the extracted subtitles Parquet dataset to."),
) -> None:
    """Extract subtitles from a `.srt`/`.ass` file or a video file into a Parquet dataset."""
    suffix = input_path.suffix.lower()
    extractor_cls = SUBTITLE_EXTRACTOR_BY_SUFFIX.get(suffix)

    if extractor_cls is not None:
        extractor = extractor_cls.from_file(input_path)
        result_path = extractor.extract(output_path)
    else:
        stream_idx = _select_stream_idx(input_path)
        extractor = VideoSubtitleExtractor.from_file(input_path)
        result_path = extractor.extract(output_path, stream_idx=stream_idx)

    typer.echo(f"Extracted subtitles written to {result_path}")


if __name__ == "__main__":
    app()
