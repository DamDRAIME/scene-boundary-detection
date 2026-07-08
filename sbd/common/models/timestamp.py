from dataclasses import dataclass
from datetime import timedelta


@dataclass
class Timestamps:
    start: timedelta
    end: timedelta

    @property
    def mid(self):
        return self.start + ((self.end - self.start) / 2)

    @property
    def duration(self):
        return self.end - self.start

    def first_half(self) -> "Timestamps":
        return Timestamps(self.start, self.mid)

    def second_half(self) -> "Timestamps":
        return Timestamps(self.mid, self.end)

    def __repr__(self) -> str:
        return f"{str(self.start)} --> {str(self.end)}"
