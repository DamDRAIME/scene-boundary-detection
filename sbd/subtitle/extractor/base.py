from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py

from sbd.subtitle.extractor.exceptions import SubtitleExtractionError
from sbd.subtitle.extractor.filehandler.base import SubtitleFileHandler
from sbd.subtitle.extractor.filehandler.models import SubTitle


class SubtitleExtractor(ABC):
    def __init__(self, filehandler: SubtitleFileHandler):
        self.filehandler = filehandler

    def extract(self, output_filepath: str | Path, **iter_subtitles_kwargs) -> Path:
        output_filepath = Path(output_filepath).with_suffix(".h5")

        with self.hdf5_datasets(output_filepath) as (subtitles, timestamps):
            subtitle_idx = -1
            for subtitle_idx, subtitle in enumerate(self.iter_subtitles(**iter_subtitles_kwargs)):
                subtitles.resize(subtitle_idx + 1, axis=0)
                subtitles[subtitle_idx] = subtitle.content
                timestamps.resize(subtitle_idx + 1, axis=0)
                timestamps[subtitle_idx] = subtitle.timestamp.start.total_seconds()

            subtitles.attrs["source"] = str(self.filehandler.filepath)
            subtitles.attrs["n_subtitles"] = subtitle_idx + 1
            timestamps.attrs["unit"] = "seconds"

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path, **kwargs) -> "SubtitleExtractor":
        pass

    def iter_subtitles(self, *args, **kwargs) -> Iterator[SubTitle]:
        yield from self.filehandler.iter_subtitles(*args, **kwargs)

    @contextmanager
    def hdf5_datasets(self, output_filepath: str | Path):
        h5_fh = h5py.File(output_filepath, "w")
        variable_length_string_dtype = h5py.string_dtype(encoding="utf-8", length=None)
        subtitles = h5_fh.create_dataset("subtitles", shape=(0,), maxshape=(None,), dtype=variable_length_string_dtype)
        timestamps = h5_fh.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="float64")
        try:
            yield subtitles, timestamps
        except Exception as e:
            raise SubtitleExtractionError("An error occurred at the creation of the HDF5 dataset.") from e
        finally:
            h5_fh.close()
