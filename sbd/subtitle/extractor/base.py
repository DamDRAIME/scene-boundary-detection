from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Self

import polars as pl

from sbd.common.models.parquet_dataset import write_parquet
from sbd.exceptions import SubtitleExtractionError
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.models import SubTitle


class SubtitleExtractor(ABC):
    _subtitle_schema = {
        "idx": pl.Int64,
        "filepath": pl.Utf8,
        "line_idx": pl.Int64,
        "timestamp_start": pl.Float64,
        "timestamp_end": pl.Float64,
        "content": pl.Utf8,
        "x1": pl.Int64,
        "y1": pl.Int64,
        "x2": pl.Int64,
        "y2": pl.Int64,
    }

    def __init__(self, filehandler: SubtitleFileHandler, data_type: str = "subtitle"):
        self.filehandler = filehandler
        self.data_type = data_type

    def extract(self, output_filepath: str | Path, **iter_subtitles_kwargs) -> Path:
        rows = [subtitle.serialize() for subtitle in self.iter_subtitles(**iter_subtitles_kwargs)]
        df = pl.DataFrame(rows, schema=self._subtitle_schema)
        metadata = {
            "type": self.data_type,
            "source": str(self.filehandler.filepath),
            "n_entries": str(len(df)),
        }
        try:
            return write_parquet(output_filepath, df, metadata)
        except Exception as e:
            raise SubtitleExtractionError("An error occurred at the creation of the Parquet dataset.") from e

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        pass

    def iter_subtitles(self, *args, **kwargs) -> Iterator[SubTitle]:
        yield from self.filehandler.iter_subtitles(*args, **kwargs)
