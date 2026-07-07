from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py

from sbd.sprite.extractor.exceptions import SpriteExtractionError
from sbd.sprite.extractor.filehandler.base import SpriteFileHandler
from sbd.sprite.extractor.filehandler.models import SpriteImg


class SpriteExtractor(ABC):
    def __init__(self, filehandler: SpriteFileHandler):
        self.filehandler = filehandler

    def extract(
        self,
        output_filepath: str | Path,
        dataset_attributes: list[str] = ["width", "height", "fps", "mode", "method"],
        **iter_sprites_kwargs,
    ) -> Path:
        output_filepath = Path(output_filepath).with_suffix(".h5")

        with self.hdf5_datasets(output_filepath) as (data, timestamps):
            sprite_idx = -1
            for sprite_idx, (timestamp, sprite) in enumerate(self.iter_sprites(**iter_sprites_kwargs)):
                data.resize(sprite_idx + 1, axis=0)
                data[sprite_idx] = sprite
                timestamps.resize(sprite_idx + 1, axis=0)
                timestamps[sprite_idx] = timestamp

            data.attrs["type"] = "sprite"
            data.attrs["source"] = str(self.filehandler.filepath)
            data.attrs["n_entries"] = sprite_idx + 1
            for attr_name in dataset_attributes:
                data.attrs[attr_name] = getattr(self, attr_name, "N/A")
            timestamps.attrs["unit"] = "seconds"

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> "SpriteExtractor":
        pass

    def iter_sprites(self, *args, **kwargs) -> Iterator[tuple[float, SpriteImg]]:
        yield from self.filehandler.iter_sprites(*args, **kwargs)

    @contextmanager
    def hdf5_datasets(self, output_filepath: str | Path):
        h5_fh = h5py.File(output_filepath, "w")
        data = h5_fh.create_dataset(
            "data",
            shape=(0, self.height, self.width, 3),
            maxshape=(None, self.height, self.width, 3),
            dtype="uint8",
            chunks=(1, self.height, self.width, 3),  # Chunked by sprite for performance at retrieval
        )
        timestamps = h5_fh.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="float64")
        try:
            yield data, timestamps
        except Exception as e:
            raise SpriteExtractionError("An error occurred at the creation of the HDF5 dataset.") from e
        finally:
            h5_fh.close()
