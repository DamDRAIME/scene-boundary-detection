from pathlib import Path

from sbd.shared.models.hdf5_dataset import HDF5TimestampedImgDataset


class SpriteDataset(HDF5TimestampedImgDataset):
    def __init__(self, filepath: str | Path):
        super().__init__(filepath, "sprites")
