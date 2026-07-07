from pathlib import Path

from sbd.shared.models.hdf5_dataset import HDF5TimestampedStrDataset


class SubtitleDataset(HDF5TimestampedStrDataset):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)


class UtteranceDataset(HDF5TimestampedStrDataset):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
