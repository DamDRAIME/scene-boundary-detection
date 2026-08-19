from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py

from sbd.exceptions import ShotExtractionError
from sbd.shot.extractor.filehandler.base import ShotFileHandler
from sbd.shot.extractor.filehandler.models import FrameImg, Shot


class ShotExtractor(ABC):
    def __init__(self, filehandler: ShotFileHandler):
        self.filehandler = filehandler

    def extract(
        self,
        output_filepath: str | Path,
        dataset_attributes: list[str] = ["width", "height"],
        **iter_shots_kwargs,
    ) -> Path:
        output_filepath = Path(output_filepath).with_suffix(".h5")

        with self.hdf5_datasets(output_filepath) as (data, timestamps, scores):
            shot_idx = -1
            for shot_idx, shot in enumerate(self.iter_shots(**iter_shots_kwargs)):
                data.resize(shot_idx + 1, axis=0)
                data[shot_idx] = shot.frame
                timestamps.resize(shot_idx + 1, axis=0)
                timestamps[shot_idx] = shot.timestamp
                scores.resize(shot_idx + 1, axis=0)
                scores[shot_idx] = shot.score if shot.score is not None else float("nan")

            data.attrs["type"] = "sprite"
            data.attrs["source"] = str(self.filehandler.filepath)
            data.attrs["n_entries"] = shot_idx + 1
            for attr_name in dataset_attributes:
                data.attrs[attr_name] = getattr(self, attr_name, "N/A")
            timestamps.attrs["unit"] = "seconds"

        return output_filepath

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> "ShotExtractor":
        pass

    @abstractmethod
    def iter_shots(self, *args, **kwargs) -> Iterator[Shot]:
        yield from self.filehandler.iter_shots(*args, **kwargs)

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
        scores = h5_fh.create_dataset("score", shape=(0,), maxshape=(None,), dtype="float64")
        try:
            yield data, timestamps, scores
        except Exception as e:
            raise ShotExtractionError("An error occurred at the creation of the HDF5 dataset.") from e
        finally:
            h5_fh.close()
