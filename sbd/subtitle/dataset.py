from pathlib import Path
from typing import Self

import polars as pl

from sbd.shared.models.parquet_dataset import ParquetTimestampedDataset, write_parquet
from sbd.subtitle.models import SubTitle, Utterance
from sbd.subtitle.utterance import subtitles_to_utterances


class SubtitleDataset(ParquetTimestampedDataset[SubTitle]):
    def _row_to_obj(self, row: dict) -> SubTitle:
        return SubTitle.deserialize(row)

    def to_utterance_dataset(self, output_filepath: str | Path, **kwargs) -> "UtteranceDataset":
        subtitles = [self._row_to_obj(row) for row in self._df.iter_rows(named=True)]
        utterances = subtitles_to_utterances(subtitles, **kwargs)
        return UtteranceDataset.from_utterances(
            utterances, output_filepath, source=self.metadata.get("source", str(self.filepath))
        )


class UtteranceDataset(ParquetTimestampedDataset[Utterance]):
    _utterance_schema = {
        "idx": pl.Int64,
        "timestamp_start": pl.Float64,
        "timestamp_end": pl.Float64,
        "content": pl.Utf8,
        "filepath": pl.Utf8,
        "line_idxs": pl.List(pl.Int64),
    }

    def _row_to_obj(self, row: dict) -> Utterance:
        return Utterance.deserialize(row)

    @classmethod
    def from_utterances(cls, utterances: list[Utterance], output_filepath: str | Path, source: str = "") -> Self:
        rows = [u.serialize() for u in utterances]
        df = pl.DataFrame(rows, schema=cls._utterance_schema)
        metadata = {"type": "utterance", "source": source, "n_entries": str(len(df))}
        output_filepath = write_parquet(output_filepath, df, metadata)
        return cls(output_filepath)
