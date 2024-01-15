"""Library internal representation for a temporal document."""

from typing import List, Literal, Tuple, Union

from .common import Instance, Span


class Event(Span):
    pass


class TimeExpression(Span):
    pass


TemporalDocEntType = Literal["event", "timex3"]
# features all realtion-entity types
TemporalRelType = Literal[
    "IDENTITY",
    "DURING",
    "DURING_INV",
    "SIMULTANEOUS",
    "BEFORE",
    "AFTER",
    "INCLUDES",
    "IS_INCLUDED",
    "IAFTER",
    "IBEFORE",
    "BEGINS",
    "ENDS",
    "BEGUN_BY",
    "ENDED_BY",
    "CYCLE",
    "IBCYCLE",
    "IACYCLE",
    "CYCLEIB",
    "CYCLEIA",
]
# features only realtion-entity types with unique point-relation
TemporalNormRelType = Literal[
    "IDENTITY",
    "BEFORE",
    "AFTER",
    "INCLUDES",
    "IS_INCLUDED",
    "IAFTER",
    "IBEFORE",
    "BEGINS",
    "ENDS",
    "BEGUN_BY",
    "ENDED_BY",
    "CYCLE",
    "IBCYCLE",
    "IACYCLE",
    "CYCLEIB",
    "CYCLEIA",
]
# features only forward realtion-entity types
TemporalMinRelType = Literal[
    "IDENTITY",
    "BEFORE",
    "INCLUDES",
    "IBEFORE",
    "BEGINS",
    "ENDS",
    "CYCLE",
    "IBCYCLE",
    "IACYCLE",
]


class TemporalRelation(Instance):
    src: Tuple[TemporalDocEntType, int]
    tgt: Tuple[TemporalDocEntType, int]
    rel_type: TemporalRelType


class TemporalDoc:
    """Representation of a temporal document recognized by our temporal model

    Constructed from
        `TmlSimpleTemporalDoc`
    Used by:
        `TemporalDocReader`

    Stores the following information
        name: name or index of the document
        text: text content of the document
        events: event spans in the document
        timeExps: time expression spans in the document
        t_relations: relations between the spans in the document
    """

    def __init__(
        self,
        name: Union[int, str],
        text: str,
        events: List[Event],
        timeExps: List[TimeExpression],
        t_relations: List[TemporalRelation],
    ):
        self.name = name
        self.text = text
        self.events = events
        self.timeExps = timeExps
        self.t_relations = t_relations
