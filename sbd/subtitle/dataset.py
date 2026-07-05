import bisect
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5py

from sbd.subtitle.exceptions import SubtitleParsingError
from sbd.subtitle.utils import convert_to_seconds


class SubtitleDataset:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Input file not found: {str(self.filepath)}")
        if not self.filepath.suffix == ".h5":
            raise ValueError(f"Input file must be an HDF5 file with .h5 extension: {str(self.filepath)}")
        self.metadata = self._load_metadata()
        self._timestamps = self._load_timestamps()
        self._subtitles = None  # Lazy loading of subtitles

    def __len__(self):
        return self.metadata["subtitles"].get("n_subtitles", len(self._timestamps))

    def __getitem__(self, key) -> tuple[float, str] | list[tuple[float, str]]:
        if self._subtitles is None:
            self._subtitles = self._load_subtitles()
        timestamps = self._timestamps[key]
        subtitles = self._subtitles[key]
        return (timestamps, subtitles) if isinstance(key, int) else list(zip(timestamps, subtitles))

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

    @contextmanager
    def open(self):
        try:
            h5_fh = h5py.File(self.filepath, "r")
            yield h5_fh
        except Exception as e:
            raise SubtitleParsingError("An error occurred while opening the HDF5 dataset.") from e
        finally:
            h5_fh.close()

    def _load_metadata(self) -> dict[dict[str, Any]]:
        metadata = defaultdict(dict)
        with self.open() as h5_fh:
            root_datasets = [key for key in h5_fh.keys() if isinstance(h5_fh[key], h5py.Dataset)]
            for dataset_name in root_datasets:
                dataset = h5_fh
            for key, val in dataset.attrs.items():
                metadata[dataset_name][key] = val
        return metadata

    def _load_timestamps(self) -> list[float]:
        with self.open() as h5_fh:
            if "timestamps" not in h5_fh:
                raise SubtitleParsingError(f"'timestamps' dataset not found in HDF5 file: {str(self.filepath)}")
            return h5_fh["timestamps"][:]

    def _load_subtitles(self) -> list[str]:
        with self.open() as h5_fh:
            if "subtitles" not in h5_fh:
                raise SubtitleParsingError(f"'subtitles' dataset not found in HDF5 file: {str(self.filepath)}")
            return h5_fh["subtitles"].asstr()[:]
