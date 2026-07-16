from pathlib import Path
from typing import Optional

import typer
from InquirerPy import get_style, inquirer
from InquirerPy.base.control import Choice

from sbd.sprite.extractor.extractors import MHTMLSpriteExtractor, VideoSpriteExtractor
from sbd.sprite.extractor.filehandler.models import ExtractionMethod
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
    subtitle_streams = VideoFileHandler.get_source_metadata(filepath)
    if len(subtitle_streams) == 1:
        return subtitle_streams[0].index

    codec_width = max(len(stream.codec_name) for stream in subtitle_streams)
    lang_width = max(len(stream.language or "und") for stream in subtitle_streams)
    choices = [
        Choice(
            value=stream.index,
            name=f"[#{stream.index:<3}] {stream.codec_name.upper():<{codec_width}}  "
            f"{(stream.language or 'und').upper():<{lang_width}}  {stream.title or '(untitled)'}",
        )
        for stream in subtitle_streams
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


@app.command()
def extract_sprites(
    input_path: Path = typer.Argument(..., exists=True, help="Path to a video or `.mhtml` file."),
    output_path: Path = typer.Argument(..., help="Path to write the extracted sprites HDF5 dataset to."),
    fps: float = typer.Option(1.0, help="Number of sprites to extract per second. Video files only."),
    height: Optional[int] = typer.Option(
        None, help="Resize height. Mutually exclusive with `--scale-ratio`. Video files only."
    ),
    width: Optional[int] = typer.Option(
        None, help="Resize width. Mutually exclusive with `--scale-ratio`. Video files only."
    ),
    scale_ratio: Optional[float] = typer.Option(
        None, help="Rescaling ratio to apply. Mutually exclusive with `--height`/`--width`. Video files only."
    ),
    method: ExtractionMethod = typer.Option(
        ExtractionMethod.SELECT.value, help="Frame extraction method to use. Video files only."
    ),
    grid_rows: int = typer.Option(3, help="Sprite sheet grid rows. `.mhtml` files only."),
    grid_cols: int = typer.Option(3, help="Sprite sheet grid columns. `.mhtml` files only."),
) -> None:
    """Extract sprites from a video or `.mhtml` sprite-sheet file into an HDF5 dataset."""
    if input_path.suffix.lower() == ".mhtml":
        extractor = MHTMLSpriteExtractor.from_file(input_path, grid_shape=(grid_rows, grid_cols))
        result_path = extractor.extract(output_path)
    else:
        extractor = VideoSpriteExtractor.from_file(input_path)
        result_path = extractor.extract(
            output_path, fps=fps, height=height, width=width, scale_ratio=scale_ratio, method=method
        )

    typer.echo(f"Extracted sprites written to {result_path}")


if __name__ == "__main__":
    app()
