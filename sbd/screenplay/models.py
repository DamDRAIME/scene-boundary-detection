from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Label(StrEnum):
    A = "Act"
    C = "Character"
    D = "Deletion"
    E = "Extension"
    G = "Camera Guidance"
    I = "Character Introduction"
    M = "Metadata"
    N = "Narrative"
    O = "Omit"
    P = "Parenthetical"
    S = "Scene Heading"
    T = "Transition"
    U = "Utterance"
    Y = "Chyron"


@dataclass
class ScreenplayElement:
    id: int
    source_line_start_idx: int
    source_line_stop_idx: int
    recently_modified: bool
    _type: Label | str

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "type": self._type.value if isinstance(self._type, Label) else self._type,
            "source_line_start_idx": self.source_line_start_idx,
            "source_line_stop_idx": self.source_line_stop_idx,
            "recently_modified": self.recently_modified,
        }
        if hasattr(self, "value"):
            data["value"] = self.value
        elif hasattr(self, "content"):
            data["content"] = [c.to_dict() for c in self.content]
        return data


@dataclass
class ScreenplayChildElement(ScreenplayElement):
    value: str


@dataclass
class ScreenplayParentElement(ScreenplayElement):
    content: list["ScreenplayParentElement", ScreenplayChildElement]


@dataclass(kw_only=True)
class Utterance(ScreenplayChildElement):
    _type: Label = Label.U


@dataclass(kw_only=True)
class Parenthetical(ScreenplayChildElement):
    _type: Label = Label.P


@dataclass(kw_only=True)
class Narrative(ScreenplayChildElement):
    _type: Label = Label.N


@dataclass(kw_only=True)
class Transition(ScreenplayChildElement):
    _type: Label = Label.T


@dataclass(kw_only=True)
class Metadata(ScreenplayChildElement):
    _type: Label = Label.M


@dataclass(kw_only=True)
class Chyron(ScreenplayChildElement):
    _type: Label = Label.Y


@dataclass(kw_only=True)
class Dialogue(ScreenplayParentElement):
    character: str
    extensions: list[str]
    content: list[Utterance | Parenthetical] = field(default_factory=list)
    same_time_as: int | None = None
    _type: str = "Dialogue"

    def to_dict(self) -> dict:
        data = super().to_dict()
        data |= {"character": self.character, "same_time_as": self.same_time_as, "extensions": self.extensions}
        return data


@dataclass(kw_only=True)
class Scene(ScreenplayParentElement):
    id: int
    heading: str
    content: list[Narrative | Transition | Metadata | Dialogue | Chyron] = field(default_factory=list)
    deleted: bool = False
    act: str | None = None
    _type: Label = Label.S

    def to_dict(self) -> dict:
        data = super().to_dict()
        data |= {"deleted": self.deleted, "heading": self.heading, "act": self.act}
        return data


@dataclass
class ParsedLine:
    line_idx: int  # Index starts at 1
    labels: frozenset[Label]
    content: str
    labels_priority: list[Label] = field(default_factory=list)

    @property
    def sanitized_content(self) -> str:
        return self.content.rstrip().rstrip("*").rstrip().lstrip()

    @property
    def is_stared(self) -> bool:
        return self.content.rstrip().endswith("*")

    @property
    def is_scene_boundary(self) -> bool:
        return any(scene_label in self.labels for scene_label in [Label.S, Label.D])

    @property
    def primary_label(self) -> Label:
        if self.labels_priority is None:
            return next(iter(self.labels), Label.O)
        for p in self.labels_priority:
            if p in self.labels:
                return p
        return next(iter(self.labels), Label.O)

    def get_primary_label(self, priority: list[Label] | None = None) -> Label:
        self.labels_priority = priority
        return self.primary_label
