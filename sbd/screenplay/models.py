from __future__ import annotations

from dataclasses import dataclass, field

from sbd.screenplay.typings import Label


@dataclass
class ScreenplayElement:
    id: int
    line_start_idx: int
    line_stop_idx: int
    _type: Label | str = None

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "type": self._type.value if isinstance(self._type, Label) else self._type,
            "value": self.value,
            "line_start_idx": self.line_start,
            "line_stop_idx": self.line_stop,
        }
        if hasattr(self, "value"):
            data["value"] = self.value
        elif hasattr(self, "content"):
            data["content"] = [c.to_dict() for c in self.content]
        return data


@dataclass
class Utterance(ScreenplayElement):
    value: str
    _type: Label = Label.U


@dataclass
class Parenthetical(ScreenplayElement):
    value: str
    _type: Label = Label.P


@dataclass
class Narrative(ScreenplayElement):
    value: str
    _type: Label = Label.N


@dataclass
class Transition(ScreenplayElement):
    value: str
    _type: Label = Label.T


@dataclass
class Metadata(ScreenplayElement):
    value: str
    _type: Label = Label.M


@dataclass
class Dialogue(ScreenplayElement):
    character: str
    extensions: list[str]
    content: list[Utterance | Parenthetical] = field(default_factory=list)
    same_time_as: int | None = None
    _type: str = "Dialogue"

    def to_dict(self) -> dict:
        data = super().to_dict()
        data |= {"character": self.character, "same_time_as": self.same_time_as, "extensions": self.extensions}
        return data


@dataclass
class Scene(ScreenplayElement):
    id: int
    heading: str
    content: list[Narrative | Transition | Metadata | Dialogue] = field(default_factory=list)
    deleted: bool = False
    _type: Label = Label.S

    def to_dict(self) -> dict:
        data = super().to_dict()
        data |= {"deleted": self.deleted, "heading": self.heading}
        return data


@dataclass
class ParsedLine:
    line_idx: int  # Index starts at 1
    labels: frozenset[Label]
    content: str

    @property
    def sanitized_content(self):
        return self.content.strip()

    def get_primary_label(self, priority: list[Label]) -> Label:
        for p in priority:
            if p in self.labels:
                return p
        return next(iter(self.labels), Label.O)
