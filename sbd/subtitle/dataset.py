from sbd.common.models.parquet_dataset import ParquetTimestampedDataset
from sbd.subtitle.models import SubTitle, Utterance


class SubtitleDataset(ParquetTimestampedDataset[SubTitle]):
    def _row_to_obj(self, row: dict) -> SubTitle:
        return SubTitle.deserialize(row)


class UtteranceDataset(ParquetTimestampedDataset[Utterance]):
    def _row_to_obj(self, row: dict) -> Utterance:
        return Utterance.deserialize(row)
