import bisect
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Generic, TypeVar

import polars as pl

from sbd.common.utils.timedelta import convert_to_seconds

T = TypeVar("T")


def write_parquet(filepath: str | Path, df: pl.DataFrame, metadata: dict[str, str]) -> Path:
    filepath = Path(filepath).with_suffix(".parquet")
    df.write_parquet(filepath, metadata={key: str(val) for key, val in metadata.items()})
    return filepath


class ParquetTimestampedDataset(ABC, Generic[T]):
    timestamp_column = "timestamp_start"

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {str(self.filepath)}")
        if self.filepath.suffix != ".parquet":
            raise ValueError(f"Input file must be a Parquet file with .parquet extension: {str(self.filepath)}")
        self.metadata = self._load_metadata()
        self._df = pl.read_parquet(self.filepath)
        self._timestamps = self._df[self.timestamp_column].to_numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(filepath={self.filepath.name}, metadata={self.metadata})"

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key: int | slice) -> tuple[float, T] | list[tuple[float, T]]:
        if isinstance(key, int):
            return self._timestamps[key], self._row_to_obj(self._df.row(key, named=True))
        timestamps = self._timestamps[key]
        rows = self._df[key].iter_rows(named=True)
        return [(ts, self._row_to_obj(row)) for ts, row in zip(timestamps, rows)]

    def find_lt(self, timestamp: float | timedelta) -> int:
        """Find rightmost value less than timestamp"""
        timestamp = convert_to_seconds(timestamp)
        idx = bisect.bisect_left(self._timestamps, timestamp)
        if idx:
            return idx - 1
        raise ValueError

    def find_le(self, timestamp: float | timedelta) -> int:
        """Find rightmost value less than or equal to timestamp"""
        timestamp = convert_to_seconds(timestamp)
        idx = bisect.bisect_right(self._timestamps, timestamp)
        if idx:
            return idx - 1
        raise ValueError

    def find_gt(self, timestamp: float | timedelta) -> int:
        """Find leftmost value greater than timestamp"""
        timestamp = convert_to_seconds(timestamp)
        idx = bisect.bisect_right(self._timestamps, timestamp)
        if idx != len(self):
            return idx
        raise ValueError

    def find_ge(self, timestamp: float | timedelta) -> int:
        """Find leftmost item greater than or equal to x"""
        timestamp = convert_to_seconds(timestamp)
        idx = bisect.bisect_left(self._timestamps, timestamp)
        if idx != len(self):
            return idx
        raise ValueError

    def find_nearest(self, timestamp: float | timedelta) -> int:
        """Find the index of the entry with the nearest timestamp to the given timestamp."""
        timestamp = convert_to_seconds(timestamp)
        idx = bisect.bisect_left(self._timestamps, timestamp)
        if idx == 0:
            return 0
        if idx == len(self):
            return len(self) - 1
        before = self._timestamps[idx - 1]
        after = self._timestamps[idx]
        if after - timestamp < timestamp - before:
            return idx
        else:
            return idx - 1

    def find_between(self, start: float | timedelta, end: float | timedelta) -> tuple[int, int]:
        """Find all entry indices between start and end timestamps (inclusive)."""
        start, end = convert_to_seconds(start), convert_to_seconds(end)
        if start > end:
            raise ValueError("`start` timestamp must be less than or equal to `end` timestamp.")
        return self.find_ge(start), self.find_le(end)

    def _load_metadata(self) -> dict:
        metadata = pl.read_parquet_metadata(self.filepath)
        return {key: val for key, val in metadata.items() if not key.startswith("ARROW:")}

    @abstractmethod
    def _row_to_obj(self, row: dict) -> T:
        pass
