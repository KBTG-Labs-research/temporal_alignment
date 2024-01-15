"""Raw TimeML module provide facilites for a parsed TimeML document and coversion tooling
to this library internal representation for a temporal document.
"""
from typing import Dict, List, Literal, Set, TextIO, Tuple, Union

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from pydantic import BaseModel

from .common import Instance, Span
from .temporal_doc import Event, TemporalDoc, TemporalRelation, TimeExpression


class TmlEvent(Span):
    """Represents a TimeML Event span in the TimeML document."""

    pass


class TmlTimeX3(Span):
    """Represents a TimeML TimeX3 span in the TimeML document."""

    pass


class TmlEventInstance(Instance):
    """Represents a TimeML EventInstance in the TimeML document.
    Unlike an `TmlEvent`, `TmlEventInstance` represents an abstract occurrence of the event as mentioned by the document, as opposed to the actual text refering to the event.

    For instance, if a text mentions a particular event happening multiple times, the corresponding `TmlEvent` would be refered to multiple `TmlEventInstance`.
    """

    event_id: str


# NOTE: "event" refers to event instances
TmlTLinkElemType = Literal["event", "timex3"]
TmlTLinkRelType = Literal[
    "BEFORE",
    "AFTER",
    "INCLUDES",
    "IS_INCLUDED",
    "DURING",
    "DURING_INV",
    "SIMULTANEOUS",
    "IAFTER",
    "IBEFORE",
    "IDENTITY",
    "BEGINS",
    "ENDS",
    "BEGUN_BY",
    "ENDED_BY",
]
tlink_rel_types: Set[TmlTLinkRelType] = {
    "SIMULTANEOUS",
    "BEFORE",
    "AFTER",
    "IBEFORE",
    "IAFTER",
    "INCLUDES",
    "IS_INCLUDED",
    "DURING",
    "DURING_INV",
    "BEGINS",
    "BEGUN_BY",
    "ENDS",
    "ENDED_BY",
    "IDENTITY",
}
TmlTLinkExtRelType = Union[TmlTLinkRelType, Literal["OVERLAP"]]
tlink_ext_rel_types: Set[TmlTLinkExtRelType] = tlink_rel_types.union({"OVERLAP"})

TmlALinkRelType = Literal[
    "CONTINUES", "CULMINATES", "INITIATES", "REINITIATES", "TERMINATES"
]
alink_rel_types: Set[TmlALinkRelType] = {
    "CONTINUES",
    "CULMINATES",
    "INITIATES",
    "REINITIATES",
    "TERMINATES",
}
TmlSLinkRelType = Literal[
    "CONDITIONAL", "COUNTER_FACTIVE", "EVIDENTIAL", "FACTIVE", "MODAL", "NEG_EVIDENTIAL"
]
slink_rel_types: Set[TmlSLinkRelType] = {
    "CONDITIONAL",
    "COUNTER_FACTIVE",
    "EVIDENTIAL",
    "FACTIVE",
    "MODAL",
    "NEG_EVIDENTIAL",
}


class TLink(Instance):
    src_type: TmlTLinkElemType
    tgt_type: TmlTLinkElemType
    src_id: str
    tgt_id: str
    rel_type: TmlTLinkExtRelType


class ConvertionIssues(BaseModel):
    doc: str
    type: str
    element: str


class TmlSimpleTemporalDoc:
    """TimeML Simplified Temporal Relation Document

    NOTE: "simlified" as in it does not parse everything defined by TimeML
    and will only cover parts related to our Temporal-Entity relation task
    """

    def __init__(
        self,
        name: str,
        text: str,
        events: Dict[str, TmlEvent],
        timex3s: Dict[str, TmlTimeX3],
        event_instances: Dict[str, TmlEventInstance],
        tlinks: Dict[str, TLink],
        issues: List[ConvertionIssues],
    ):
        self.name = name
        self.text = text
        self.events = events
        self.timex3s = timex3s
        self.event_instances = event_instances
        self.tlinks = tlinks
        self.issues = issues

    @classmethod
    def from_xml(
        cls,
        name: str,
        file_or_text: Union[str, TextIO],
        parse_dct=True,
        convert_overlap2during=True,
    ):
        """Construct TimeML Simple Temproal Document from TimeML XML file

        Args:
            name (str): name of document
            file_or_text (Union[str, TextIO]): XML file to be parse
            parse_dct (bool, optional): parse document creation time as first token. Defaults to True.
            convert_overlap2during (bool, optional): covert OVERLAP annotation into DURING. OVERLAP is used by some TimeML dataset but is not an actual valid TimeML TLINK relation type. Defaults to True.
        """
        text = ""
        events: Dict[str, TmlEvent] = {}
        timex3s: Dict[str, TmlTimeX3] = {}
        event_instances: Dict[str, TmlEventInstance] = {}
        tlinks: Dict[str, TLink] = {}
        issues: List[ConvertionIssues] = []

        # [1] parse document
        soup = BeautifulSoup(file_or_text, "lxml-xml")

        # [1.1] parse DCT
        if parse_dct:
            dtc_timex3 = soup.find("DCT").find("TIMEX3")  # type: ignore
            assert isinstance(dtc_timex3, Tag)
            assert dtc_timex3.name == "TIMEX3"
            assert len(dtc_timex3.contents) == 1
            assert isinstance(dtc_timex3.contents[0], NavigableString)
            ins_id = dtc_timex3["tid"]
            assert isinstance(ins_id, str)
            assert ins_id not in timex3s
            timex3s[ins_id] = TmlTimeX3(
                id_=ins_id,
                start=0,
                end=0,
            )

        # [1.2] parse text
        text_elem = soup.find("TEXT")
        assert isinstance(text_elem, Tag)

        for elem in text_elem.contents:
            if isinstance(elem, NavigableString):
                text += str(elem)
            elif isinstance(elem, Tag):
                if elem.name == "EVENT":
                    assert len(elem.contents) == 1
                    assert isinstance(elem.contents[0], NavigableString)
                    start = len(text)
                    text += str(elem.contents[0])
                    end = len(text)
                    ins_id = elem["eid"]
                    assert isinstance(ins_id, str)
                    assert ins_id not in events
                    events[ins_id] = TmlEvent(
                        id_=ins_id,
                        start=start,
                        end=end,
                    )
                elif elem.name == "TIMEX3":
                    assert len(elem.contents) == 1
                    assert isinstance(elem.contents[0], NavigableString)
                    start = len(text)
                    text += str(elem.contents[0])
                    end = len(text)
                    ins_id = elem["tid"]
                    assert isinstance(ins_id, str)
                    assert ins_id not in timex3s
                    timex3s[ins_id] = TmlTimeX3(
                        id_=ins_id,
                        start=start,
                        end=end,
                    )
                else:
                    assert elem.name in {"SIGNAL"}
            else:
                assert False

        # [1.3] parse instances
        for event_instance in soup.find_all("MAKEINSTANCE"):
            ins_id = event_instance["eiid"]
            assert isinstance(ins_id, str)
            assert ins_id not in timex3s
            event_instances[ins_id] = TmlEventInstance(
                id_=ins_id,
                event_id=event_instance["eventID"],
            )
        for tlink in soup.find_all("TLINK"):
            if "timeID" in tlink.attrs:
                src_type = "timex3"
                src_id = tlink["timeID"]
                assert src_id in timex3s
            else:
                src_type = "event"
                src_id = tlink["eventInstanceID"]
                assert src_id in event_instances
            if "relatedToTime" in tlink.attrs:
                tgt_type = "timex3"
                tgt_id = tlink["relatedToTime"]
                assert tgt_id in timex3s
            else:
                tgt_type = "event"
                tgt_id = tlink["relatedToEventInstance"]
                assert tgt_id in event_instances
            ins_id = tlink["lid"]
            assert isinstance(ins_id, str)
            assert ins_id not in tlinks
            tlink_rel_type = tlink["relType"]

            if convert_overlap2during and tlink_rel_type == "OVERLAP":
                tlink_rel_type = "DURING"
                issues.append(
                    ConvertionIssues(
                        doc=name,
                        type=f"TLINK of 'OVERLAP' relation converted to 'DURING'",
                        element=ins_id,
                    )
                )

            if tlink_rel_type not in tlink_ext_rel_types:
                if tlink_rel_type in slink_rel_types:
                    issues.append(
                        ConvertionIssues(
                            doc=name, type="SLINK being passed as TLINK", element=ins_id
                        )
                    )
                elif tlink_rel_type in alink_rel_types:
                    issues.append(
                        ConvertionIssues(
                            doc=name, type="ALINK being passed as TLINK", element=ins_id
                        )
                    )
                else:
                    issues.append(
                        ConvertionIssues(
                            doc=name,
                            type=f"TLINK of unknown relation: '{tlink_rel_type}'",
                            element=ins_id,
                        )
                    )
                continue
            tlinks[ins_id] = TLink(
                id_=ins_id,
                src_type=src_type,
                tgt_type=tgt_type,
                src_id=src_id,
                tgt_id=tgt_id,
                rel_type=tlink_rel_type,
            )

        return cls(
            name=name,
            text=text,
            events=events,
            timex3s=timex3s,
            event_instances=event_instances,
            tlinks=tlinks,
            issues=issues,
        )

    def to_new_temproal_doc(self) -> Tuple[TemporalDoc, List[ConvertionIssues]]:
        events: List[Event] = []
        timeExps: List[TimeExpression] = []
        t_relations: List[TemporalRelation] = []
        issues: List[ConvertionIssues] = []

        # convert Events
        eventsId2Idx: Dict[str, int] = {}

        for idx, (ins_id, event) in enumerate(self.events.items()):
            eventsId2Idx[ins_id] = idx
            events.append(Event(**event.dict()))

        # create mapping for EventInstance to Event
        # NOTE: EventInstances will be remapped back to Event
        #       other issues will be handled later on
        #       issues include:
        #           1. relations of different EventInstance but of the same Event
        eventInsId2eventId: Dict[str, str] = {}
        for ins_id, event_ins in self.event_instances.items():
            eventInsId2eventId[ins_id] = event_ins.event_id

        # convert TimeX3
        timex3Id2Idx: Dict[str, int] = {}
        for idx, (ins_id, timex3) in enumerate(self.timex3s.items()):
            timex3Id2Idx[ins_id] = idx
            timeExps.append(TimeExpression(**timex3.dict()))

        for idx, (ins_id, tlink) in enumerate(self.tlinks.items()):
            if tlink.src_type == "event":
                src_idx = eventsId2Idx[eventInsId2eventId[tlink.src_id]]
            else:
                src_idx = timex3Id2Idx[tlink.src_id]
            if tlink.tgt_type == "event":
                tgt_idx = eventsId2Idx[eventInsId2eventId[tlink.tgt_id]]
            else:
                tgt_idx = timex3Id2Idx[tlink.tgt_id]

            src = (tlink.src_type, src_idx)
            tgt = (tlink.tgt_type, tgt_idx)

            if src == tgt:
                issues.append(
                    ConvertionIssues(
                        doc=self.name,
                        type="relation between different Event instances of the same Event",
                        element=tlink.id_,
                    )
                )
                continue

            tlink_rel_type = tlink.rel_type
            if tlink_rel_type == "OVERLAP":
                tlink_rel_type = "DURING"
                issues.append(
                    ConvertionIssues(
                        doc=self.name,
                        type="TLINK of 'OVERLAP' relation converted to 'DURING'",
                        element=tlink.id_,
                    )
                )

            t_relations.append(
                TemporalRelation(
                    id_=tlink.id_,
                    src=src,
                    tgt=tgt,
                    rel_type=tlink_rel_type,
                )
            )

        return TemporalDoc(self.name, self.text, events, timeExps, t_relations), issues
