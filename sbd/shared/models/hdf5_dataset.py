import bisect
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Generic, TypeVar

import h5py
import numpy as np

from sbd.shared.utils.timedelta import convert_to_seconds

T = TypeVar("T")


class HDF5TimestampedDataset(Generic[T]):
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {str(self.filepath)}")
        if self.filepath.suffix != ".h5":
            raise ValueError(f"Input file must be an HDF5 file with .h5 extension: {str(self.filepath)}")
        self.metadata = self._load_metadata()
        self._timestamps = self._load_timestamps()

    def __repr__(self):
        return f"{self.__class__.__name__}(filepath={self.filepath.name}, metadata={self.metadata['data']})"

    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, key: int | slice) -> tuple[float, T] | list[tuple[float, T]]:
        timestamps = self._timestamps[key]
        data = self._get(key)
        return (timestamps, data) if isinstance(key, int) else list(zip(timestamps, data))

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
        """Find the index of the subtitle with the nearest timestamp to the given timestamp."""
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
        """Find all subtitle indices between start and end timestamps (inclusive)."""
        start, end = convert_to_seconds(start), convert_to_seconds(end)
        if start > end:
            raise ValueError("`start` timestamp must be less than or equal to `end` timestamp.")
        return self.find_ge(start), self.find_le(end)

    def _load_metadata(self) -> dict:
        metadata = defaultdict(dict)
        with self.open() as h5_fh:
            root_datasets = [
                key for key in h5_fh.keys() if isinstance(h5_fh[key], h5py.Dataset) and key in ["data", "timestamps"]
            ]
            if not all(dataset_name in root_datasets for dataset_name in ["data", "timestamps"]):
                raise KeyError(
                    f"Required datasets 'data' and 'timestamps' not found in HDF5 file: {str(self.filepath)}"
                )
            for dataset_name in root_datasets:
                dataset = h5_fh[dataset_name]
                for key, val in dataset.attrs.items():
                    metadata[dataset_name][key] = val
        return metadata

    def _load_timestamps(self) -> list[float]:
        with self.open() as h5_fh:
            if "timestamps" not in h5_fh:
                raise KeyError(f"'timestamps' dataset not found in HDF5 file: {str(self.filepath)}")
            return h5_fh["timestamps"][:]

    def _get(self, key: int | slice) -> T | list[T]:
        with self.open() as h5_fh:
            if "data" not in h5_fh:
                raise KeyError(f"'data' dataset not found in HDF5 file: {str(self.filepath)}")
            return h5_fh["data"][key]

    @contextmanager
    def open(self) -> h5py.File:
        h5_fh = h5py.File(self.filepath, "r")
        try:
            yield h5_fh
        finally:
            h5_fh.close()


class HDF5TimestampedStrDataset(HDF5TimestampedDataset[str]):
    def _get(self, key: int | slice) -> T | list[T]:
        with self.open() as h5_fh:
            if "data" not in h5_fh:
                raise KeyError(f"'data' dataset not found in HDF5 file: {str(self.filepath)}")
            return h5_fh["data"].asstr()[key]


class HDF5TimestampedImgDataset(HDF5TimestampedDataset[np.ndarray]):
    pass
