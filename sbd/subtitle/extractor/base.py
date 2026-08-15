from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Self

import polars as pl
from sentence_transformers import SentenceTransformer

from sbd.common.models.parquet_dataset import write_parquet
from sbd.exceptions import SubtitleExtractionError
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.extractor.utterance import subtitles_to_utterances
from sbd.subtitle.models import SubTitle


class SubtitleExtractor(ABC):
    _subtitle_schema = {  # noqa: RUF012
        "idx": pl.Int64,
        "filepath": pl.Utf8,
        "line_idx": pl.Int64,
        "timestamp_start": pl.Float64,
        "timestamp_end": pl.Float64,
        "content": pl.Utf8,
        "embedding": pl.List(pl.Float64),
        "x1": pl.Int64,
        "y1": pl.Int64,
        "x2": pl.Int64,
        "y2": pl.Int64,
    }

    _utterance_schema = {  # noqa: RUF012
        "idx": pl.Int64,
        "filepath": pl.Utf8,
        "line_idxs": pl.List(pl.Int64),
        "timestamp_start": pl.Float64,
        "timestamp_end": pl.Float64,
        "content": pl.Utf8,
        "embedding": pl.List(pl.Float64),
    }

    def __init__(self, filehandler: SubtitleFileHandler, data_type: str = "subtitle"):
        self.filehandler = filehandler
        self.data_type = data_type
        self._df: pl.DataFrame | None = None

    @property
    def metadata(self) -> dict[str, str]:
        assert self._df is not None, "No subtitle data available."
        return {
            "type": self.data_type,
            "source": str(self.filehandler.filepath),
            "n_entries": str(len(self._df)),
        }

    def extract(self, **iter_subtitles_kwargs) -> Self:
        try:
            rows = [subtitle.serialize() for subtitle in self.iter_subtitles(**iter_subtitles_kwargs)]
            self._df = pl.DataFrame(rows, schema=self._subtitle_schema)
        except Exception as e:
            raise SubtitleExtractionError("An error occurred while extracting subtitles.") from e
        return self

    def compute_embeddings(self, model: SentenceTransformer) -> Self:
        assert self._df is not None, "No subtitle data available."

        try:
            embeddings = model.encode(self._df["content"].to_list())
            self._df = self._df.with_columns(pl.Series("embedding", embeddings.tolist()))
        except Exception as e:
            raise SubtitleExtractionError("An error occurred at the computation of embeddings.") from e
        return self

    def convert_to_utterances(self, **kwargs) -> Self:
        assert self._df is not None, "No subtitle data available."

        try:
            subtitles = [SubTitle.deserialize(row) for row in self._df.iter_rows(named=True)]
            utterances = subtitles_to_utterances(subtitles, **kwargs)
            rows = [u.serialize() for u in utterances]
            self._df = pl.DataFrame(rows, schema=self._utterance_schema)
            self.data_type = "utterance"
        except Exception as e:
            raise SubtitleExtractionError("An error occurred at the conversion to utterances.") from e
        return self

    def save(self, output_filepath: str | Path) -> Path:
        assert self._df is not None, "No subtitle data available."

        try:
            return write_parquet(output_filepath, self._df, self.metadata)
        except Exception as e:
            raise SubtitleExtractionError("An error occurred at the creation of the Parquet dataset.") from e

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> Self:
        pass

    def iter_subtitles(self, *args, **kwargs) -> Iterator[SubTitle]:
        yield from self.filehandler.iter_subtitles(*args, **kwargs)
