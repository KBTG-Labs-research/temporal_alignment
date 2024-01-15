from __future__ import annotations

import heapq
from itertools import chain
from typing import (
    Counter,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from ..rep.temporal_doc import (
    TemporalDoc,
    TemporalDocEntType,
    TemporalMinRelType,
    TemporalNormRelType,
    TemporalRelation,
    TemporalRelType,
)

RealPointRelation = Literal["BEFORE", "AFTER", "SAME"]
PointRelation = Literal["BEFORE", "AFTER", "SAME", "UNK", "AMBIGUOUS"]
MinPointRelationWA = Literal["BEFORE", "AFTER", "SAME", "AMBIGUOUS"]
# Minimal point-relation
# Unknown relation should be represented with a lack of relation
# After relation should be represented with the inverse (Before).
MinPointRelation = Literal["BEFORE", "SAME"]

# Entity relation definition in terms of PointRelations among start & end points
# SRC_start TGT_start
# SRC_start TGT_end
# SRC_end   TGT_start
# SRC_end   TGT_end
PointRelationDef = Tuple[PointRelation, PointRelation, PointRelation, PointRelation]

NodeName = str
PointType = Literal["S", "E"]
PointName = Tuple[NodeName, PointType]

# TemporalRelTypeWC = Union[TemporalRelType, Literal["CYCLE", "IBCYCLE", "IACYCLE"]]
# TemporalNormRelTypeWC = Union[
#     TemporalNormRelType, Literal["CYCLE", "IBCYCLE", "IACYCLE"]
# ]
TemporalRelTypeWU = Union[TemporalRelType, Literal["UNK"]]
TemporalNormRelTypeWU = Union[TemporalNormRelType, Literal["UNK"]]
# TemporalMinRelTypeWUnk = Union[TemporalMinRelType, Literal["UNK"]]
TimeMlRelation = Tuple[NodeName, NodeName, TemporalRelType]
TimeMlRelationWU = Tuple[NodeName, NodeName, TemporalRelTypeWU]
TimeMlMinRelation = Tuple[NodeName, NodeName, TemporalMinRelType]

TmlPointRelation = Tuple[PointName, PointName, PointRelation]
TmlMinPointRelationWA = Tuple[PointName, PointName, MinPointRelationWA]
TmlMinPointRelation = Tuple[PointName, PointName, MinPointRelation]

# mapping for entity-relation to point relations
TML_RELATION_TO_POINT_RELATION: Dict[TemporalRelType, PointRelationDef] = {
    "IDENTITY": ("SAME", "BEFORE", "AFTER", "SAME"),
    "SIMULTANEOUS": ("SAME", "BEFORE", "AFTER", "SAME"),
    "DURING": ("SAME", "BEFORE", "AFTER", "SAME"),
    "DURING_INV": ("SAME", "BEFORE", "AFTER", "SAME"),
    "BEFORE": ("BEFORE", "BEFORE", "BEFORE", "BEFORE"),
    "AFTER": ("AFTER", "AFTER", "AFTER", "AFTER"),
    "IBEFORE": ("BEFORE", "BEFORE", "SAME", "BEFORE"),
    "IAFTER": ("AFTER", "SAME", "AFTER", "AFTER"),
    "BEGINS": ("SAME", "BEFORE", "AFTER", "BEFORE"),
    "BEGUN_BY": ("SAME", "BEFORE", "AFTER", "AFTER"),
    "ENDS": ("AFTER", "BEFORE", "AFTER", "SAME"),
    "ENDED_BY": ("BEFORE", "BEFORE", "AFTER", "SAME"),
    "IS_INCLUDED": ("AFTER", "BEFORE", "AFTER", "BEFORE"),
    "INCLUDES": ("BEFORE", "BEFORE", "AFTER", "AFTER"),
    "CYCLE": ("AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS"),
    "IBCYCLE": ("BEFORE", "BEFORE", "AMBIGUOUS", "AMBIGUOUS"),
    "IACYCLE": ("AMBIGUOUS", "AMBIGUOUS", "AFTER", "AFTER"),
    "CYCLEIB": ("AFTER", "AMBIGUOUS", "AFTER", "AMBIGUOUS"),
    "CYCLEIA": ("AMBIGUOUS", "BEFORE", "AMBIGUOUS", "BEFORE"),
}
# mapping for point-relations to entity-relation
TML_POINT_RELATION_TO_RELATION: Dict[PointRelationDef, TemporalNormRelType] = {
    ("SAME", "BEFORE", "AFTER", "SAME"): "IDENTITY",
    ("BEFORE", "BEFORE", "BEFORE", "BEFORE"): "BEFORE",
    ("AFTER", "AFTER", "AFTER", "AFTER"): "AFTER",
    ("BEFORE", "BEFORE", "SAME", "BEFORE"): "IBEFORE",
    ("AFTER", "SAME", "AFTER", "AFTER"): "IAFTER",
    ("SAME", "BEFORE", "AFTER", "BEFORE"): "BEGINS",
    ("SAME", "BEFORE", "AFTER", "AFTER"): "BEGUN_BY",
    ("AFTER", "BEFORE", "AFTER", "SAME"): "ENDS",
    ("BEFORE", "BEFORE", "AFTER", "SAME"): "ENDED_BY",
    ("AFTER", "BEFORE", "AFTER", "BEFORE"): "IS_INCLUDED",
    ("BEFORE", "BEFORE", "AFTER", "AFTER"): "INCLUDES",
    ("AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS"): "CYCLE",
    ("BEFORE", "BEFORE", "AMBIGUOUS", "AMBIGUOUS"): "IBCYCLE",
    ("AMBIGUOUS", "AMBIGUOUS", "AFTER", "AFTER"): "IACYCLE",
    ("AFTER", "AMBIGUOUS", "AFTER", "AMBIGUOUS"): "CYCLEIB",
    ("AMBIGUOUS", "BEFORE", "AMBIGUOUS", "BEFORE"): "CYCLEIA",
}
# mapping for normalizing temporal relations
TML_RELATION_NORM: Dict[TemporalRelTypeWU, TemporalNormRelTypeWU] = {
    "IDENTITY": "IDENTITY",
    "SIMULTANEOUS": "IDENTITY",
    "DURING": "IDENTITY",
    "DURING_INV": "IDENTITY",
    "BEFORE": "BEFORE",
    "AFTER": "AFTER",
    "IBEFORE": "IBEFORE",
    "IAFTER": "IAFTER",
    "BEGINS": "BEGINS",
    "BEGUN_BY": "BEGUN_BY",
    "ENDS": "ENDS",
    "ENDED_BY": "ENDED_BY",
    "IS_INCLUDED": "IS_INCLUDED",
    "INCLUDES": "INCLUDES",
    "UNK": "UNK",
    "CYCLE": "CYCLE",
    "IBCYCLE": "IBCYCLE",
    "IACYCLE": "IACYCLE",
    "CYCLEIB": "CYCLEIB",
    "CYCLEIA": "CYCLEIA",
}
# mapping for normalizing temporal relations
TML_RELATION_INV: Dict[TemporalNormRelType, TemporalMinRelType] = {
    "AFTER": "BEFORE",
    "IS_INCLUDED": "INCLUDES",
    "IAFTER": "IBEFORE",
    "BEGUN_BY": "BEGINS",
    "ENDED_BY": "ENDS",
    "CYCLEIB": "IBCYCLE",
    "CYCLEIA": "IACYCLE",
}
# ("S", "E", "BEFORE")
# means point relation
# As BEFORE Be
PointRelationStructure = Tuple[PointType, PointType, PointRelation]
# mapping for entity-relation to informational point-relations
TML_RELATION_TO_POINT_INFO: Dict[TemporalNormRelType, List[PointRelationStructure]] = {
    "IDENTITY": [("S", "S", "SAME"), ("E", "E", "SAME")],
    "BEFORE": [("E", "S", "BEFORE")],
    "INCLUDES": [("S", "S", "BEFORE"), ("E", "E", "AFTER")],
    "IBEFORE": [("E", "S", "SAME")],
    "BEGINS": [("S", "S", "SAME"), ("E", "E", "BEFORE")],
    "ENDS": [("S", "S", "AFTER"), ("E", "E", "SAME")],
    "AFTER": [("S", "E", "AFTER")],
    "IS_INCLUDED": [("S", "S", "AFTER"), ("E", "E", "BEFORE")],
    "IAFTER": [("S", "E", "SAME")],
    "BEGUN_BY": [("S", "S", "SAME"), ("E", "E", "AFTER")],
    "ENDED_BY": [("S", "S", "BEFORE"), ("E", "E", "SAME")],
    "CYCLE": [
        ("S", "S", "AMBIGUOUS"),
        ("S", "E", "AMBIGUOUS"),
        ("E", "E", "AMBIGUOUS"),
    ],
    "IBCYCLE": [
        ("S", "S", "BEFORE"),
        ("E", "S", "AMBIGUOUS"),
    ],
    "IACYCLE": [
        ("S", "E", "AMBIGUOUS"),
        ("E", "E", "AFTER"),
    ],
    "CYCLEIB": [
        ("S", "S", "AFTER"),
        ("S", "E", "AMBIGUOUS"),
    ],
    "CYCLEIA": [
        ("E", "S", "AMBIGUOUS"),
        ("E", "E", "BEFORE"),
    ],
}


class TmlGraphPoint:
    def __init__(self, name: NodeName, point_type: PointType) -> None:
        self.name: NodeName = name
        self.type: PointType = point_type
        self.same: Set[PointName] = {
            (self.name, self.type),
        }
        self.before: Set[PointName] = set()
        self.after: Set[PointName] = set()

    def compute_ambiguous(self) -> Set[PointName]:
        """returns a set of point which have ambiguous relation with `self`

        NOTE: Return an empty set if no points have ambiguous relation with `self`
              However, if it has ambiguous with some point, it will also have relation with `self`
        """
        return (
            self.before.intersection(self.after)
            | self.same.intersection(self.after)
            | self.same.intersection(self.before)
        )

    def get_point_relation_with(self, point: PointName) -> PointRelation:
        """Get point relation of self compared to another point

        NOTE: this method does not work with violation

        Args:
            point (PointName): another point

        Returns:
            PointRelation: point relation
        """
        if point in self.compute_ambiguous():
            return "AMBIGUOUS"
        elif point in self.before:
            # if point is BEFORE self, then self is after point
            return "AFTER"
        elif point in self.same:
            # point is SAME as self
            return "SAME"
        elif point in self.after:
            # if point is AFTER self, then self is BEFORE point
            return "BEFORE"
        return "UNK"

    def check_point_relation_with(self, point: PointName, rel: PointRelation) -> bool:
        act_rel = self.get_point_relation_with(point)
        if act_rel == "AMBIGUOUS":
            return True
        else:
            return rel == act_rel

    def relation_count(self):
        return len(self.before) + len(self.same) + len(self.after)

    def has_closure_violation(self):
        return (
            len(self.same.intersection(self.before)) > 0
            or len(self.same.intersection(self.after)) > 0
            or len(self.before.intersection(self.after)) > 0
        )

    def clone(self) -> "TmlGraphPoint":
        other = TmlGraphPoint(self.name, self.type)
        other.same.update(self.same)
        other.before.update(self.before)
        other.after.update(self.after)
        return other


def to_min_point_rel_wa(point_rel: TmlPointRelation) -> TmlMinPointRelationWA:
    """Convert TmlPointRelation to MinPointRelation

    which essentially converts 'AFTER' point relation into 'BEFORE'"""
    src, tgt, rel = point_rel
    if rel == "AFTER":
        return tgt, src, "BEFORE"
    if rel == "UNK":
        raise ValueError(
            "point relation (TmlPointRelation) UNK cannot be converted to TmlMinPointRelation"
        )
    return src, tgt, rel


# def to_min_point_rel(point_rel: TmlPointRelation) -> TmlMinPointRelation:
#     """Convert TmlPointRelation to MinPointRelation

#     which essentially converts 'AFTER' point relation into 'BEFORE'"""
#     src, tgt, rel = point_rel
#     if rel == "AFTER":
#         return tgt, src, "BEFORE"
#     if rel == "UNK":
#         raise ValueError(
#             "point relation (TmlPointRelation) UNK cannot be converted to TmlMinPointRelation"
#         )
#     if rel == "AMBIGUOUS":
#         raise ValueError(
#             "point relation (TmlPointRelation) AMBIGUOUS cannot be converted to TmlMinPointRelation"
#         )
#     return src, tgt, rel


def to_min_ent_rel(point_rel: TimeMlRelation) -> TimeMlMinRelation:
    """Convert TmlPointRelation to MinPointRelation

    which essentially converts 'AFTER' point relation into 'BEFORE'"""
    src, tgt, rel = point_rel
    rel = TML_RELATION_NORM[rel]
    if rel == "UNK":
        raise ValueError(
            "point relation (TmlPointRelation) UNK cannot be converted to TmlMinPointRelation"
        )
    if rel in TML_RELATION_INV:
        return tgt, src, TML_RELATION_INV[rel]
    rel = cast(TemporalMinRelType, rel)
    if rel == "IDENTITY" or rel == "AMBIGUOUS":
        return *sorted((src, tgt)), rel
    return src, tgt, rel


class PointCluster:
    """PointCluster represent points in a PointGraph which are at the
    exact same time (cluster).

    PointCluster does not mutate over the course of the search
    """

    # index of PointCluster in PointGraph
    idx: int
    # Name of point cluster, which will be the name of some point in the cluster
    name: PointName
    # Points in this cluster
    points: Set[PointName]
    # NOTE: order of the point cluster are insignificant
    # Set of PointCluster(s) which are immediately next to this PointCluster
    next: Set[PointCluster]
    # Set of PointCluster(s) which are immediately previous to this PointCluster
    prev: Set[PointCluster]
    is_amb: bool

    def __init__(
        self,
        name: PointName,
        points: Set[PointName],
        next: Set[PointCluster],
        prev: Set[PointCluster],
        is_amb: bool,
    ) -> None:
        self.idx = 0
        self.name = name
        self.points = points
        self.next = next
        self.prev = prev
        self.is_amb = is_amb

    def __hash__(self) -> int:
        return hash(self.name)

    def __ior__(self, other: PointCluster):
        self.points |= other.points
        for other_next in other.next:
            other_next.prev.remove(other)
            other_next.prev.add(self)
        self.next |= other.next
        for other_prev in other.prev:
            other_prev.next.remove(other)
            other_prev.next.add(self)
        self.prev |= other.prev
        assert (
            self.is_amb == other.is_amb
            or (not self.is_amb and len(self.points) == 1)
            or (not other.is_amb and len(other.points) == 1)
        )
        self.is_amb = self.is_amb or other.is_amb
        return self

    def iterate_point_pair(self) -> Generator[Tuple[PointName, PointName], None, None]:
        points = sorted(self.points)
        for idx, src in enumerate(points):
            for tgt in points[idx + 1 :]:
                yield src, tgt

    def amb_should_be(self, is_amb: bool):
        assert (
            is_amb or not self.is_amb
        ), "if should not be ambiguous, then should should not already be marked as ambiguous."
        self.is_amb = self.is_amb or is_amb


# `Chain` does not mutate over the course of the search
# NOTE: order of the point cluster are insignificant
class PointGraph:
    clusters: List[PointCluster]
    pt_2_cluster: Dict[PointName, PointCluster]
    entities: Set[str]
    # list entity with inferable point relations
    entity_inf_pt_rels: List[str]

    def __init__(
        self,
        # core point relations with entity inferable relations
        core_pt_w_ent_inf_rels: List[TmlMinPointRelationWA],
    ):
        self.clusters = []
        self.pt_2_cluster: Dict[PointName, PointCluster] = {}
        self.entities: Set[str] = set()
        self.entity_inf_pt_rels = []

        for src, tgt, pt_rel in core_pt_w_ent_inf_rels:
            self.entities.add(src[0])
            self.entities.add(tgt[0])

            # is_amb = pt_rel == "AMBIGUOUS"
            src_pc = self.get_point_cluster(src)
            tgt_pc = self.get_point_cluster(tgt)
            if pt_rel == "BEFORE":
                src_pc.next.add(tgt_pc)
                tgt_pc.prev.add(src_pc)
                if src[0] == tgt[0] and src[1] == "S" and tgt[1] == "E":
                    self.entity_inf_pt_rels.append(src[0])
            else:
                assert (
                    pt_rel == "SAME" or pt_rel == "AMBIGUOUS"
                ), f"Core relation must not have UNK relation"
                # join clusters
                src_pc |= tgt_pc
                self.pt_2_cluster[tgt] = src_pc
                self.clusters.remove(tgt_pc)
                src_pc.amb_should_be(pt_rel == "AMBIGUOUS")

        # run clusters' index number
        for idx, cluster in enumerate(self.clusters):
            cluster.idx = idx

    def get_point_cluster(self, pt: PointName):
        if pt in self.pt_2_cluster:
            cluster = self.pt_2_cluster[pt]
            return cluster
        pc = PointCluster(pt, {pt}, set(), set(), False)
        self.pt_2_cluster[pt] = pc
        self.clusters.append(pc)
        return pc

    def count_total_pt_rel(self) -> Tuple[int, int]:
        same_count = 0
        before_count = 0
        for cluster in self.clusters:
            same_count += len(cluster.points) - 1
            before_count += len(cluster.next)
        return same_count, before_count


class PointClusterState:
    forest: Dict[PointName, Set[PointName]]
    tree_count: int
    # read-only next PointCluster
    ro_next: Set[PointCluster]

    def __init__(
        self,
        forest: Dict[PointName, Set[PointName]],
        tree_count: int,
        next: Set[PointCluster],
        # prev: Set[PointCluster],
    ):
        self.forest = forest
        self.tree_count = tree_count
        self.ro_next = next
        # self.prev = prev

    def is_in_same_tree(self, a: PointName, b: PointName) -> bool:
        return self.forest[a] == self.forest[b]

    def join(self, a: PointName, b: PointName) -> bool:
        tree_a = self.forest[a]
        tree_b = self.forest[b]
        if tree_a == tree_b:
            return False
        tree = tree_a | tree_b
        # clone forest
        self.forest = dict(self.forest)
        for node in tree:
            self.forest[node] = tree
        self.tree_count -= 1
        return True

    def add_next(self, pc: PointCluster):
        self.ro_next = {*self.ro_next, pc}

    # def add_prev(self, pc: PointCluster):
    #     self.prev = {*self.prev, pc}

    def clone(self) -> PointClusterState:
        return PointClusterState(
            forest=self.forest,
            tree_count=self.tree_count,
            next=self.ro_next,
            # prev=self.prev,
        )

    @classmethod
    def from_point_cluster(cls, pt_c: PointCluster):
        return cls(
            forest={point: {point} for point in pt_c.points},
            tree_count=len(pt_c.points),
            next=set(),
            # prev=set(),
        )


ChainStatePos = Tuple[Literal["SAME"], int, int, int]


class ChainState:
    pt_graph: PointGraph
    pc_states: List[PointClusterState]
    pending_same_pt_rel_count: int
    pending_before_pt_rel_count: int
    redundant_pt_rel_counter: Counter[str]
    pt_rels: List[TmlMinPointRelationWA]
    ent_rels: List[TimeMlRelation]
    last_pos: ChainStatePos

    def __init__(
        self,
        pt_graph: PointGraph,
        pc_states: List[PointClusterState],
        pending_same_pt_rel_count: int,
        pending_before_pt_rel_count: int,
        # redundant_pt_rel_counter: Counter[str],
        redundant_pt_rel_count: int,
        pt_rels: List[TmlMinPointRelationWA],
        ent_rels: List[TimeMlRelation],
        last_pos: ChainStatePos,
    ):
        # reference to PointGraph
        self.pt_graph = pt_graph
        # state of each point cluster
        self.pc_states = pc_states
        # number of pending point relations
        assert (
            pending_same_pt_rel_count >= 0
        ), "pending_same_pt_rel_count must have value >= 0"
        self.pending_same_pt_rel_count = pending_same_pt_rel_count
        assert (
            pending_before_pt_rel_count >= 0
        ), "pending_before_pt_rel_count must have value >= 0"
        self.pending_before_pt_rel_count = pending_before_pt_rel_count
        # number of redundant point relations
        # self.redundant_pt_rel_counter = redundant_pt_rel_counter
        self.redundant_pt_rel_count = redundant_pt_rel_count
        # point relations inferable from entity relations (and entity spans)
        self.pt_rels = pt_rels
        # entity relations added to state
        self.ent_rels = ent_rels
        # state last search position
        self.last_pos = last_pos

    @classmethod
    def from_point_graph(cls, pt_graph: PointGraph):
        (
            pending_same_pt_rel_count,
            pending_before_pt_rel_count,
        ) = pt_graph.count_total_pt_rel()
        pc_states: List[PointClusterState] = [
            PointClusterState.from_point_cluster(cluster)
            for cluster in pt_graph.clusters
        ]
        pt_rels: List[TmlMinPointRelationWA] = []
        ent_rels: List[TimeMlRelation] = []
        # infer all entity span relations
        for ent in pt_graph.entity_inf_pt_rels:
            src_pc = pt_graph.get_point_cluster((ent, "S"))
            tgt_pc = pt_graph.get_point_cluster((ent, "E"))
            # directly add to next, to modify in-place
            pc_states[src_pc.idx].ro_next.add(tgt_pc)
            pt_rels.append(((ent, "S"), (ent, "E"), "BEFORE"))
            pending_before_pt_rel_count -= 1
        # infer all identity relations
        # iterate over clusters to identify identity relations
        for cluster in pt_graph.clusters:
            if cluster.is_amb:
                # iterate over point pairs in the cluster
                for s_src_pt, s_tgt_pt in cluster.iterate_point_pair():
                    if s_src_pt[1] != "S" or s_tgt_pt[1] != "S":
                        continue
                    # only look at pairs of starting points

                    # get cluster for start points
                    s_src_c = pt_graph.get_point_cluster(s_src_pt)
                    s_tgt_c = pt_graph.get_point_cluster(s_tgt_pt)
                    # get cluster for end points
                    e_src_pt: PointName = s_src_pt[0], "E"
                    e_tgt_pt: PointName = s_tgt_pt[0], "E"
                    e_src_c = pt_graph.get_point_cluster(e_src_pt)
                    e_tgt_c = pt_graph.get_point_cluster(e_tgt_pt)

                    # if end points are in the same cluster and source are not already in the same tree within the state
                    # NOTE: Example, in join A, B, C
                    #       We will vist (A, B), (A, C), (B, C).
                    #       In such case, we will already join A, B, C with the first two
                    #       visits. And thus, will not have to join when we visit (B, C)
                    #       As a result, the `is_in_same_tree` check is required to not
                    #       add redundant IDENTITY relations.
                    if (
                        s_src_c == s_tgt_c
                        and s_src_c == e_src_c
                        and e_src_c == e_tgt_c
                        and not pc_states[cluster.idx].is_in_same_tree(
                            s_src_pt, s_tgt_pt
                        )
                    ):
                        # NOTE: CYCLE pairs assumption:
                        # "SAME" points relations are given in pairs.
                        # Therefore, if starting points are not in the same tree,
                        # then the ending points must also not be in the same tree.
                        # This assumption allows greedy identification of CYCLE relations
                        assert not pc_states[e_src_c.idx].is_in_same_tree(
                            e_src_pt, e_tgt_pt
                        ), "violatation of CYCLE pairs assumption"

                        pt_rels.append((s_src_pt, e_tgt_pt, "AMBIGUOUS"))
                        pc_states[cluster.idx].join(s_src_pt, s_tgt_pt)
                        pending_same_pt_rel_count -= 1
                        if not pc_states[cluster.idx].is_in_same_tree(
                            s_src_pt, e_src_pt
                        ):
                            pt_rels.append((s_src_pt, e_src_pt, "AMBIGUOUS"))
                            pc_states[cluster.idx].join(s_src_pt, e_src_pt)
                            pending_same_pt_rel_count -= 1

                        if not pc_states[cluster.idx].is_in_same_tree(
                            s_tgt_pt, e_tgt_pt
                        ):
                            pt_rels.append((e_tgt_pt, e_tgt_pt, "AMBIGUOUS"))
                            pc_states[cluster.idx].join(s_tgt_pt, e_tgt_pt)
                            pending_same_pt_rel_count -= 1
                        ent_rels.append((*sorted((s_src_pt[0], s_tgt_pt[0])), "CYCLE"))
            else:
                # iterate over point pairs in the cluster
                for s_src_pt, s_tgt_pt in cluster.iterate_point_pair():
                    if s_src_pt[1] != "S" or s_tgt_pt[1] != "S":
                        continue
                    # only look at pairs of starting points

                    # get cluster for end points
                    e_src_pt: PointName = s_src_pt[0], "E"
                    e_tgt_pt: PointName = s_tgt_pt[0], "E"
                    e_src_c = pt_graph.get_point_cluster(e_src_pt)
                    e_tgt_c = pt_graph.get_point_cluster(e_tgt_pt)

                    # if end points are in the same cluster and source are not already in the same tree within the state
                    # NOTE: Example, in join A, B, C
                    #       We will vist (A, B), (A, C), (B, C).
                    #       In such case, we will already join A, B, C with the first two
                    #       visits. And thus, will not have to join when we visit (B, C)
                    #       As a result, the `is_in_same_tree` check is required to not
                    #       add redundant IDENTITY relations.
                    if e_src_c == e_tgt_c and not pc_states[
                        cluster.idx
                    ].is_in_same_tree(s_src_pt, s_tgt_pt):
                        # NOTE: IDENTITY pairs assumption:
                        # "SAME" points relations are given in pairs.
                        # Therefore, if starting points are not in the same tree,
                        # then the ending points must also not be in the same tree.
                        # This assumption allows greedy identification of IDENITY relations
                        assert not pc_states[e_src_c.idx].is_in_same_tree(
                            e_src_pt, e_tgt_pt
                        ), "violatation of IDENTITY pairs assumption"

                        pt_rels.append((s_src_pt, s_tgt_pt, "SAME"))
                        pt_rels.append((e_src_pt, e_tgt_pt, "SAME"))
                        pc_states[cluster.idx].join(s_src_pt, s_tgt_pt)
                        pc_states[e_src_c.idx].join(e_src_pt, e_tgt_pt)
                        pending_same_pt_rel_count -= 2
                        ent_rels.append(
                            (*sorted((s_src_pt[0], s_tgt_pt[0])), "IDENTITY")
                        )
        return cls(
            pt_graph=pt_graph,
            pc_states=pc_states,
            pending_same_pt_rel_count=pending_same_pt_rel_count,
            pending_before_pt_rel_count=pending_before_pt_rel_count,
            # redundant_pt_rel_counter=Counter(),
            redundant_pt_rel_count=0,
            pt_rels=pt_rels,
            ent_rels=ent_rels,
            last_pos=("SAME", -1, 0, 0),
        )

    def clone(self):
        return ChainState(
            pt_graph=self.pt_graph,
            pc_states=[pc_state.clone() for pc_state in self.pc_states],
            pending_same_pt_rel_count=self.pending_same_pt_rel_count,
            pending_before_pt_rel_count=self.pending_before_pt_rel_count,
            # redundant_pt_rel_counter=Counter(self.redundant_pt_rel_counter),
            redundant_pt_rel_count=self.redundant_pt_rel_count,
            pt_rels=list(self.pt_rels),
            ent_rels=list(self.ent_rels),
            last_pos=self.last_pos,
        )

    def add_point_relation(
        self, src_pt: PointName, tgt_pt: PointName, rel: MinPointRelationWA
    ) -> bool:
        src_cluster = self.pt_graph.get_point_cluster(src_pt)
        src_pc = self.pc_states[src_cluster.idx]
        reduced_pending = False
        if rel == "SAME" or rel == "AMBIGUOUS":
            if src_pc.join(src_pt, tgt_pt):
                self.pt_rels.append((src_pt, tgt_pt, rel))
                assert self.pending_same_pt_rel_count > 0
                self.pending_same_pt_rel_count -= 1
                reduced_pending = True
            else:
                # self.redundant_pt_rel_counter["SAME redundant"] += 1
                self.redundant_pt_rel_count += 1
        else:
            assert rel == "BEFORE"
            tgt_cluster = self.pt_graph.get_point_cluster(tgt_pt)
            if tgt_cluster in src_cluster.next:
                if tgt_cluster not in src_pc.ro_next:
                    src_pc.add_next(tgt_cluster)
                    self.pt_rels.append((src_pt, tgt_pt, rel))
                    self.pending_before_pt_rel_count -= 1
                    reduced_pending = True
                else:
                    # self.redundant_pt_rel_counter["BEFORE redundant"] += 1
                    self.redundant_pt_rel_count += 1
            else:
                # self.redundant_pt_rel_counter["BEFORE not closest"] += 1
                self.redundant_pt_rel_count += 1
        return reduced_pending

    def add_entity_relation(self, src: str, tgt: str, rel: TemporalNormRelType) -> bool:
        reduced_pending = False
        # add corresponding point-relation in minimal form
        for src_type, tgt_type, pt_rel in TML_RELATION_TO_POINT_INFO[rel]:
            if self.add_point_relation(
                *to_min_point_rel_wa(((src, src_type), (tgt, tgt_type), pt_rel))
            ):
                reduced_pending = True
        # add entity-relation in minimal form
        self.ent_rels.append(to_min_ent_rel((src, tgt, rel)))
        return reduced_pending

    def __lt__(self, other: ChainState) -> bool:
        s = (
            # sum(self.redundant_pt_rel_counter.values()),
            # self.redundant_pt_rel_count,
            len(self.ent_rels),
            self.pending_same_pt_rel_count,
            self.pending_before_pt_rel_count,
        )
        o = (
            # sum(other.redundant_pt_rel_counter.values()),
            # other.redundant_pt_rel_count,
            len(other.ent_rels),
            other.pending_same_pt_rel_count,
            other.pending_before_pt_rel_count,
        )
        return s < o


class TmlGraph:
    def __init__(self):
        self.points: Dict[
            PointName,
            TmlGraphPoint,
        ] = {}

    def add_node(self, name: NodeName):
        assert (name, "S") not in self.points and (
            name,
            "E",
        ) not in self.points, f"node with name {name} already exist in the graph"

        begin = TmlGraphPoint(name, "S")
        end = TmlGraphPoint(name, "E")
        self.points[name, "S"] = begin
        self.points[name, "E"] = end

        self.add_point_relation(begin, end, "BEFORE")

        return begin, end

    def relation_count(self) -> int:
        return sum(point.relation_count() for point in self.points.values())

    def has_closure_violation(self) -> bool:
        return any(point.has_closure_violation() for point in self.points.values())

    def add_point_relation(
        self,
        src_point: TmlGraphPoint,
        tgt_point: TmlGraphPoint,
        point_rel: PointRelation,
    ) -> None:
        """Add relation between two points

        Args:
            src_point (TmlGraphPoint): source point
            tgt_point (TmlGraphPoint): target point
            point_rel (PointRelation): relation
        """
        src_before = list(src_point.before)
        src_same = list(src_point.same)
        src_after = list(src_point.after)
        tgt_before = list(tgt_point.before)
        tgt_same = list(tgt_point.same)
        tgt_after = list(tgt_point.after)

        if point_rel == "SAME":
            for s in src_same:
                # source points inherit all three relation from the target points
                src = self.points[s]
                src.before.update(tgt_before)
                src.same.update(tgt_same)
                src.after.update(tgt_after)
            for s in src_before:
                src = self.points[s]
                src.after.update(tgt_same)
                src.after.update(tgt_after)
            for s in src_after:
                src = self.points[s]
                src.before.update(tgt_before)
                src.before.update(tgt_same)

            for t in tgt_same:
                # target points inherit all three relation from the source points
                tgt = self.points[t]
                tgt.before.update(src_before)
                tgt.same.update(src_same)
                tgt.after.update(src_after)
            for t in tgt_before:
                tgt = self.points[t]
                tgt.after.update(src_same)
                tgt.after.update(src_after)
            for t in tgt_after:
                tgt = self.points[t]
                tgt.before.update(src_before)
                tgt.before.update(src_same)

        elif point_rel == "BEFORE":
            for s in chain(src_same, src_before):
                # source points inherit all same and after relation from the target points as after relation
                src = self.points[s]
                src.after.update(tgt_same)
                src.after.update(tgt_after)

            for t in chain(tgt_same, tgt_after):
                # target points inherit all before and same relation from the source points as before relation
                tgt = self.points[t]
                tgt.before.update(src_before)
                tgt.before.update(src_same)

        else:
            assert point_rel == "AFTER", f"point_rel: got {point_rel} expect AFTER"

            for s in chain(src_same, src_after):
                # source points inherit all before and same relation from the target points as before relation
                src = self.points[s]
                src.before.update(tgt_before)
                src.before.update(tgt_same)

            for t in chain(tgt_same, tgt_before):
                # target points inherit all same and after relation from the source points as after relation
                tgt = self.points[t]
                tgt.after.update(src_same)
                tgt.after.update(src_after)

    def safe_add_relations(
        self,
        relations: List[TimeMlRelation],
    ) -> Tuple[List[TimeMlRelation], List[TimeMlRelation], List[TimeMlRelation]]:
        """Safely add relations

        NOTE: this method returns the TE3 non-redundant relations, which are not acutally non-redundant relations
              to acutally get the non-reducant relations, use `get_non_redundant_relations` method

        Returns:
            List[TemporalRelation]: TE3 non-redundant, TE3 redundant, closure violation
        """
        te3_non_redundant: List[TimeMlRelation] = []
        te3_redundant: List[TimeMlRelation] = []
        closure_violation: List[TimeMlRelation] = []
        # creating another graph in anticipation of a violation
        other = self.clone()
        for src, tgt, rel in relations:
            # try adding relation to the graph
            other.add_relation(src, tgt, rel, True)

            if other.has_closure_violation():
                # if the added graph result in violation
                # collect relation resulting in violation
                closure_violation.append((src, tgt, rel))
                # rollback added the relation by re-cloning self
                other = self.clone()
            else:
                # adding this relation does not result in violation
                # adding this relation to self
                # track if added relation is redudant base on current graph (greedy algorithm)
                if self.add_relation(src, tgt, rel, True):
                    te3_non_redundant.append((src, tgt, rel))
                else:
                    te3_redundant.append((src, tgt, rel))

        return te3_non_redundant, te3_redundant, closure_violation

    def get_node_points(self, node: NodeName, auto_add_node: bool):
        if auto_add_node and (node, "S") not in self.points:
            ns, ne = self.add_node(node)
        else:
            ns = self.points[node, "S"]
            ne = self.points[node, "E"]
        return ns, ne

    def get_point(self, pt: PointName, auto_add_node: bool):
        ns, ne = self.get_node_points(pt[0], auto_add_node=auto_add_node)
        return ns if pt[1] == "S" else ne

    def add_relation(
        self, src: NodeName, tgt: NodeName, rel: TemporalRelType, auto_add_node: bool
    ) -> bool:
        """Add relation between two temporal entity (node)

        Args:
            src (NodeName): source temporal entity name
            tgt (NodeName): target temporal entity name
            rel (TemporalRelType): relation type
            auto_add_node (bool): auto add new node

        Returns:
            bool: returns True if relation is novel
        """
        before_rel_count = self.relation_count()
        ss, se = self.get_node_points(src, auto_add_node=auto_add_node)
        ts, te = self.get_node_points(tgt, auto_add_node=auto_add_node)

        sa_ts, sa_te, se_ts, se_te = TML_RELATION_TO_POINT_RELATION[rel]

        self.add_point_relation(ss, ts, sa_ts)
        self.add_point_relation(ss, te, sa_te)
        self.add_point_relation(se, ts, se_ts)
        self.add_point_relation(se, te, se_te)

        after_rel_count = self.relation_count()

        return after_rel_count != before_rel_count

    def get_all_closure_relations(self) -> List[TimeMlRelation]:
        """Return a list of relations for all pairs"""
        nodes = list({node for node, _ in self.points.keys()})
        nodes.sort()
        relations: List[TimeMlRelation] = []
        for src in nodes:
            for tgt in nodes:
                if src == tgt:
                    continue

                rel = self.derive_relation(src, tgt)
                if rel != "UNK":
                    relations.append((src, tgt, rel))

        return relations

    def derive_relation(self, src: NodeName, tgt: NodeName) -> TemporalNormRelTypeWU:
        """Compute relation of the given node pair"""
        if (src, "S") not in self.points or (tgt, "S") not in self.points:
            return "UNK"

        ss = self.points[src, "S"]
        se = self.points[src, "E"]

        ss_ts = ss.get_point_relation_with((tgt, "S"))
        ss_te = ss.get_point_relation_with((tgt, "E"))
        se_ts = se.get_point_relation_with((tgt, "S"))
        se_te = se.get_point_relation_with((tgt, "E"))

        graph_point_relations = ss_ts, ss_te, se_ts, se_te

        return TML_POINT_RELATION_TO_RELATION.get(graph_point_relations, "UNK")

    def check_relation_match(
        self, src: NodeName, tgt: NodeName, rel: TemporalRelTypeWU
    ) -> bool:
        rel = TML_RELATION_NORM[rel]
        if rel == "UNK":
            UNK_AMB = ["UNK", "AMBIGUOUS"]
            return (
                self.points[src, "S"].get_point_relation_with((tgt, "S")) in UNK_AMB
                or self.points[src, "S"].get_point_relation_with((tgt, "E")) in UNK_AMB
                or self.points[src, "E"].get_point_relation_with((tgt, "S")) in UNK_AMB
                or self.points[src, "E"].get_point_relation_with((tgt, "E")) in UNK_AMB
            )
        else:
            sa_ts, sa_te, se_ts, se_te = TML_RELATION_TO_POINT_RELATION[rel]
            return (
                (src, "S") in self.points
                and self.points[src, "S"].check_point_relation_with((tgt, "S"), sa_ts)
                and self.points[src, "S"].check_point_relation_with((tgt, "E"), sa_te)
                and self.points[src, "E"].check_point_relation_with((tgt, "S"), se_ts)
                and self.points[src, "E"].check_point_relation_with((tgt, "E"), se_te)
            )

    def clone(self) -> "TmlGraph":
        """Clone TmlGraph"""
        other = self.__class__()
        for name, point in self.points.items():
            other.points[name] = point.clone()
        return other

    def compute_core_point_relations(
        self,
        keep_ent_infer_rel: bool = False,
    ) -> List[TmlMinPointRelationWA]:
        # store all core relations of the graph, including (X. S) BEFORE (X, E)
        # which is implied by entity definition
        core_rel: List[TmlMinPointRelationWA] = []
        # mark which points have been considered
        points_in_graph: Set[PointName] = set()

        # loop over each point to find core relation of each point
        for point_name in sorted(self.points.keys()):
            if point_name in points_in_graph:
                # skip points already considered (in graph), same to some point previously considered
                continue

            point_rel = self.points[point_name]
            pt_ambiguous = point_rel.compute_ambiguous()

            # collect points at the same position
            for a_pt_name in point_rel.same:
                if a_pt_name in pt_ambiguous or point_name == a_pt_name:
                    # skip itself
                    continue
                # Add same point to core relation (point name is sorted)
                core_rel.append((*sorted((point_name, a_pt_name)), "SAME"))
                # Mark point as already considered
                points_in_graph.add(a_pt_name)

            # collect core AFTER points of `point_name`
            core_after_pts: Set[PointName] = set()
            # iterate over after points to find core after points
            for a_pt_name in sorted(point_rel.after):
                if a_pt_name in pt_ambiguous:
                    continue
                a_pt_ambiguous = self.points[a_pt_name].compute_ambiguous()

                # after point is best if it is not after another after point
                # assume after point is best
                is_best_point = True
                # collect after points that are not longer best (worse)
                worse_after_pts: Set[PointName] = set()
                # check after point against current core after points
                for c_pt_name in core_after_pts:
                    if c_pt_name in a_pt_ambiguous:
                        # points in `point_rel.after` are checked in sorted order
                        # therefore, any points added prior (points in `core_after_pts`)
                        # must be better than the current `a_pt_name`
                        is_best_point = False
                    else:
                        # if after point is after or same as some current core after point
                        if (
                            a_pt_name in self.points[c_pt_name].after
                            or a_pt_name in self.points[c_pt_name].same
                        ):
                            # then it is not best after point
                            is_best_point = False
                        # if some core after point is after this after point
                        if c_pt_name in self.points[a_pt_name].after:
                            # that is is not longer a core after point and need to be removed
                            worse_after_pts.add(c_pt_name)

                if is_best_point:
                    # remove points that is worse than best point
                    core_after_pts.difference_update(worse_after_pts)
                    # add best point to after points set
                    core_after_pts.add(a_pt_name)
                else:
                    assert len(worse_after_pts) == 0

            # add core relation to graph
            for a_pt_name in core_after_pts:
                core_rel.append((point_name, a_pt_name, "BEFORE"))

            # collect points which have ambiguous relation
            if len(pt_ambiguous) > 0:
                pt_ambiguous = sorted(pt_ambiguous)
                ref_amb_point = pt_ambiguous[0]
                points_in_graph.add(ref_amb_point)
                for a_pt_name in pt_ambiguous[1:]:
                    # Add same point to core relation (point name is sorted)
                    core_rel.append((ref_amb_point, a_pt_name, "AMBIGUOUS"))
                    # Mark point as already considered
                    points_in_graph.add(a_pt_name)

        # trim inferable relations implied by entity span (X. S) BEFORE (X, E)
        trimmed_core_rel: List[TmlMinPointRelationWA] = []
        for src, tgt, rel in core_rel:
            # assumes core relation is not redundant
            inf_ent_rel: Optional[str] = None
            # iterate over all points equivalent to the source
            for pos_src in self.points[src].same:
                if pos_src[1] != "S":
                    # skip non-Start point
                    continue
                # if corresponding End point is equivalent to the target
                if (pos_src[0], "E") in self.points[tgt].same:
                    # then the relation is redudant
                    inf_ent_rel = pos_src[0]
                    break

            if inf_ent_rel is None:
                # if not inferable keep relation
                trimmed_core_rel.append((src, tgt, rel))
            elif keep_ent_infer_rel:
                # else if keep flag enabled
                # keep entity inferable relations in normalized form
                trimmed_core_rel.append(((inf_ent_rel, "S"), (inf_ent_rel, "E"), rel))

        return trimmed_core_rel

    def compute_core_entity_relations(self) -> List[TimeMlRelation]:
        core_pt_w_ent_inf_rels = self.compute_core_point_relations(True)
        point_graph = PointGraph(core_pt_w_ent_inf_rels)
        state = ChainState.from_point_graph(point_graph)
        states = [state]

        identity_nodes: Set[NodeName] = {tgt for _, tgt, _ in state.ent_rels}

        while len(states) > 0:
            state = heapq.heappop(states)  # type:ignore
            if state.pending_same_pt_rel_count != 0:
                assert state.pending_same_pt_rel_count > 0
                # there is a cluster with pending same point-relation

                # find that pending cluster
                pending_cluster: Optional[PointCluster] = None
                pending_pc_state: Optional[PointClusterState] = None
                # loop over clusters
                for cluster, pc_state in zip(point_graph.clusters, state.pc_states):
                    if pc_state.tree_count == 1:
                        # cluster has one tree, then the cluster is fully merged
                        continue
                    # found pending cluster
                    pending_cluster = cluster
                    pending_pc_state = pc_state
                    break
                assert (
                    pending_cluster != None and pending_pc_state != None
                ), f"a pending cluster must always be found since pending_same_pt_rel_count > 0"

                # iterate over all point-pairs to breath-first search over all possible relations

                points = sorted(pending_cluster.points)
                # compute start position
                if (
                    state.last_pos[0] == "SAME"
                    and pending_cluster.idx == state.last_pos[1]
                ):
                    src_begin = state.last_pos[2]
                    tgt_begin = state.last_pos[3] + 1
                else:
                    src_begin = 0
                    tgt_begin = 1

                for src_idx in range(src_begin, len(points)):
                    # compute start position

                    same_src_pt = points[src_idx]
                    found_pending_src = False
                    for tgt_idx in range(tgt_begin, len(points)):
                        same_tgt_pt = points[tgt_idx]
                        # for same_tgt_pt in points[tgt_begin:]:
                        if pending_pc_state.is_in_same_tree(same_src_pt, same_tgt_pt):
                            # src and tgt is already in the same tree
                            continue
                        rel = self.derive_relation(same_src_pt[0], same_tgt_pt[0])
                        if rel == "UNK":
                            # no valid relation that can be assigned to src and tgt
                            continue
                        next_state = state.clone()
                        reduced_pending = next_state.add_entity_relation(
                            same_src_pt[0], same_tgt_pt[0], rel
                        )
                        assert reduced_pending
                        next_state.last_pos = (
                            "SAME",
                            pending_cluster.idx,
                            src_idx,
                            tgt_idx,
                        )
                        heapq.heappush(states, next_state)  # type:ignore
                        found_pending_src = True
                    if found_pending_src:
                        break
                    # tgt_begin would be non-zero for only the first iteration in order to
                    # continue the loop of from the last position of the state
                    tgt_begin = src_idx + 2
            elif state.pending_before_pt_rel_count != 0:
                # there is a cluster with pending before point-relation

                # find that pending cluster
                pending_cluster: Optional[PointCluster] = None
                next_pending_cluster: Optional[PointCluster] = None
                pending_pc_state: Optional[PointClusterState] = None
                # loop over clusters
                for cluster, pc_state in zip(point_graph.clusters, state.pc_states):
                    pending_next = cluster.next - pc_state.ro_next
                    if len(pending_next) == 0:
                        # cluster has one tree, then the cluster is fully merged
                        continue
                    # found pending cluster
                    pending_cluster = cluster
                    pending_pc_state = pc_state
                    next_pending_cluster = next(iter(pending_next))
                    break
                assert (
                    pending_cluster is not None
                    and next_pending_cluster is not None
                    and pending_pc_state is not None
                ), f"a pending cluster must always be found since pending_same_pt_rel_count > 0"
                # iterate over all point-pairs to breath-first search over all possible relations

                # find all src, tgt to search
                pending_nodes = sorted(
                    {
                        node
                        for node, _ in pending_cluster.points
                        if node not in identity_nodes
                    }
                )
                assert (
                    len(pending_nodes) > 0
                ), f"pruning out `pending_nodes` must not leave array empty"

                next_pending_nodes = sorted(
                    {
                        node
                        for node, _ in next_pending_cluster.points
                        if node not in identity_nodes
                    }
                )
                assert (
                    len(next_pending_nodes) > 0
                ), f"pruning out `next_pending_nodes` must not leave array empty"

                for before_src in pending_nodes:
                    for before_tgt in next_pending_nodes:
                        rel = self.derive_relation(before_src, before_tgt)
                        if rel == "UNK":
                            # no valid relation that can be assigned to src and tgt
                            continue
                        next_state = state.clone()
                        reduced_pending = next_state.add_entity_relation(
                            before_src, before_tgt, rel
                        )
                        assert reduced_pending
                        heapq.heappush(states, next_state)  # type:ignore
            else:
                return state.ent_rels
        assert False, f"unreachable"


def infer_closure_relation(
    tmprlDoc: TemporalDoc, ignore_violation: bool
) -> TemporalDoc:
    """Creates a new TemporalDoc with all relations infered using graph closure."""

    event_map: Dict[str, Tuple[TemporalDocEntType, int]] = {}
    relations: List[TimeMlRelation] = []
    for relation in tmprlDoc.t_relations:
        src_t, src_i = relation.src
        tgt_t, tgt_i = relation.tgt
        src = src_t + str(src_i)
        tgt = tgt_t + str(tgt_i)

        if src not in event_map:
            event_map[src] = src_t, src_i

        if tgt not in event_map:
            event_map[tgt] = tgt_t, tgt_i

        relations.append((src, tgt, relation.rel_type))

    graph = TmlGraph()
    _, _, violations = graph.safe_add_relations(relations)

    if not ignore_violation and len(violations) > 0:
        raise ValueError("tmprlDoc contains relation(s) which violate graph closure")

    idx = 0
    infered_relations: List[TemporalRelation] = []
    for src in event_map:
        for tgt in event_map:
            if src == tgt:
                continue
            rel = graph.derive_relation(src, tgt)
            if rel != "UNK":
                infered_relations.append(
                    TemporalRelation(
                        id_=str(idx),
                        src=event_map[src],
                        tgt=event_map[tgt],
                        rel_type=rel,
                    )
                )
                idx += 1

    return TemporalDoc(
        name=tmprlDoc.name,
        text=tmprlDoc.text,
        events=tmprlDoc.events,
        timeExps=tmprlDoc.timeExps,
        # only relations are update, the rest are the same
        t_relations=infered_relations,
    )
