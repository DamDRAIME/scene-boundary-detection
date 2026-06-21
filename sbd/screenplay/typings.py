from enum import StrEnum

from sbd.screenplay.models import Dialogue, Metadata, Narrative, Transition


class Label(StrEnum):
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


SceneContent = Narrative | Transition | Metadata | Dialogue
