from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


class SpriteExtractor(ABC):
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Input file not found: {str(self.filepath)}")

    @abstractmethod
    def extract(self, output_filepath: str | Path, *args, **kwargs) -> Path:
        pass

    @abstractmethod
    def iter_sprites(self, *args, **kwargs) -> Iterator[tuple[float, np.ndarray]]:
        pass

    @contextmanager
    def hdf5_datasets(self, output_filepath: str | Path, height: int, width: int):
        h5_fh = h5py.File(output_filepath, "w")
        sprites = h5_fh.create_dataset(
            "sprites",
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            dtype="uint8",
            chunks=(1, height, width, 3),  # Chunked by sprite for performance at retrieval
        )
        timestamps = h5_fh.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="float64")
        try:
            yield sprites, timestamps
        finally:
            h5_fh.close()
