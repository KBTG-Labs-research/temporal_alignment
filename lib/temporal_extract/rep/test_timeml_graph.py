from typing import List

from .timeml_graph import (
    TemporalNormRelTypeWU,
    TimeMlRelation,
    TimeMlRelationWU,
    TmlGraph,
    TmlPointRelation,
)


def _test_adding_relations(
    relations: List[TimeMlRelation],
    expect_non_redundant: List[TimeMlRelation],
    expect_redundant: List[TimeMlRelation],
    expect_closure_violation: List[TimeMlRelation],
):
    graph = TmlGraph()
    te3_non_redundant, te3_redundant, closure_violation = graph.safe_add_relations(
        relations
    )
    assert te3_non_redundant == expect_non_redundant
    assert te3_redundant == expect_redundant
    assert closure_violation == expect_closure_violation


def test_timeml_graph_adding_relations():
    """Test added relations based on TE3 method for determining redundant relations and closure violation"""
    # base case
    _test_adding_relations(
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_non_redundant=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_redundant=[],
        expect_closure_violation=[],
    )
    # TE3 redundant
    _test_adding_relations(
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "C", "BEFORE"),
        ],
        expect_non_redundant=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_redundant=[
            ("A", "C", "BEFORE"),
        ],
        expect_closure_violation=[],
    )
    # TE3 redundant
    _test_adding_relations(
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "C", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_non_redundant=[
            ("A", "B", "BEFORE"),
            ("A", "C", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_redundant=[],
        expect_closure_violation=[],
    )
    # closure violation
    _test_adding_relations(
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "C", "AFTER"),
        ],
        expect_non_redundant=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_redundant=[],
        expect_closure_violation=[
            ("A", "C", "AFTER"),
        ],
    )


def _test_check_relation_match_no_violation(
    relations: List[TimeMlRelation],
    expect_match: List[TimeMlRelationWU],
    expect_no_match: List[TimeMlRelationWU],
):
    graph = TmlGraph()
    non_redundant, redundant, closure_violation = graph.safe_add_relations(relations)
    assert non_redundant == relations
    assert redundant == []
    assert closure_violation == []

    for src, tgt, rel in expect_match:
        assert graph.check_relation_match(src, tgt, rel)
    for src, tgt, rel in expect_no_match:
        assert not graph.check_relation_match(src, tgt, rel)


def test_timeml_check_relation_match_no_violation():
    """Test graph ability to perform relation matching"""
    _test_check_relation_match_no_violation(
        relations=[
            ("A", "B", "IDENTITY"),
        ],
        expect_match=[
            ("A", "B", "IDENTITY"),
            ("A", "B", "SIMULTANEOUS"),
            ("A", "B", "DURING"),
            ("A", "B", "DURING_INV"),
        ],
        expect_no_match=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "B", "IBEFORE"),
            ("A", "B", "IAFTER"),
            ("A", "B", "INCLUDES"),
            ("A", "B", "IS_INCLUDED"),
            ("A", "B", "BEGINS"),
            ("A", "B", "ENDS"),
            ("A", "B", "BEGUN_BY"),
            ("A", "B", "ENDED_BY"),
        ],
    )
    _test_check_relation_match_no_violation(
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_match=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "C", "BEFORE"),
            ("B", "A", "AFTER"),
            ("C", "B", "AFTER"),
            ("C", "A", "AFTER"),
        ],
        expect_no_match=[
            ("A", "B", "AFTER"),
            ("B", "C", "AFTER"),
            ("A", "C", "AFTER"),
            ("B", "A", "BEFORE"),
            ("C", "B", "BEFORE"),
            ("C", "A", "BEFORE"),
        ],
    )
    _test_check_relation_match_no_violation(
        relations=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
        expect_match=[
            ("A", "C", "BEFORE"),
            ("B", "A", "IAFTER"),
            ("C", "B", "IAFTER"),
            ("C", "A", "AFTER"),
        ],
        expect_no_match=[
            # forward
            ("A", "C", "IBEFORE"),
            ("A", "B", "IAFTER"),
            ("B", "C", "IAFTER"),
            ("A", "C", "IAFTER"),
            # backward
            ("C", "A", "IAFTER"),
            ("B", "A", "IBEFORE"),
            ("C", "B", "IBEFORE"),
            ("C", "A", "IBEFORE"),
        ],
    )

    # Test 001
    _test_check_relation_match_no_violation(
        # AAABBBBBEEE
        #    |   |
        #    |  CC
        #    \  ?/
        #     DDD
        relations=[
            ("A", "B", "IBEFORE"),
            ("A", "D", "BEFORE"),
            ("D", "E", "BEFORE"),
            ("B", "E", "IBEFORE"),
            ("C", "B", "ENDS"),
        ],
        expect_match=[
            ("A", "B", "IBEFORE"),
            ("A", "C", "BEFORE"),
            ("A", "D", "BEFORE"),
            ("A", "E", "BEFORE"),
            ("B", "A", "IAFTER"),
            ("B", "C", "ENDED_BY"),
            ("B", "D", "INCLUDES"),
            ("B", "E", "IBEFORE"),
            ("C", "A", "AFTER"),
            ("C", "B", "ENDS"),
            ("C", "D", "UNK"),
            ("C", "E", "IBEFORE"),
            ("D", "A", "AFTER"),
            ("D", "B", "IS_INCLUDED"),
            ("D", "C", "UNK"),
            ("D", "E", "BEFORE"),
        ],
        expect_no_match=[],
    )


def _test_compute_core_point_relation(
    relations: List[TimeMlRelation],
    expect_point_relation: List[TmlPointRelation],
):
    graph = TmlGraph()
    _, _, closure_violation = graph.safe_add_relations(relations)
    assert closure_violation == [], "test must not have violations"

    point_relations = graph.compute_core_point_relations()
    assert sorted(point_relations) == sorted(expect_point_relation)


def test_compute_core_point_relation():
    """Test computing core point relations"""
    _test_compute_core_point_relation(
        # A-->B
        relations=[
            ("A", "B", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A-->B
        #  \->C
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "C", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
            (("A", "E"), ("C", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A-->B-->C
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "C", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("C", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A-->B-->C
        #  \->D
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "D", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("C", "S"), "BEFORE"),
            (("A", "E"), ("D", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A-->B-->C-->E
        #  \->D-/
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "D", "BEFORE"),
            ("D", "C", "BEFORE"),
            ("C", "E", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("C", "S"), "BEFORE"),
            (("A", "E"), ("D", "S"), "BEFORE"),
            (("D", "E"), ("C", "S"), "BEFORE"),
            (("C", "E"), ("E", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A-->B-->C-->E
        #  \->D-/->F
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("A", "D", "BEFORE"),
            ("D", "C", "BEFORE"),
            ("C", "E", "BEFORE"),
            ("D", "F", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("C", "S"), "BEFORE"),
            (("A", "E"), ("D", "S"), "BEFORE"),
            (("D", "E"), ("C", "S"), "BEFORE"),
            (("D", "E"), ("F", "S"), "BEFORE"),
            (("C", "E"), ("E", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A|B
        relations=[
            ("A", "B", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
        ],
    )
    _test_compute_core_point_relation(
        # A|B|C
        relations=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
        ],
    )
    _test_compute_core_point_relation(
        # A|B|C -> D
        relations=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
            ("C", "D", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("C", "E"), ("D", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # A|B|C -> D|E
        relations=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
            ("C", "D", "BEFORE"),
            ("D", "E", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("C", "E"), ("D", "S"), "BEFORE"),
            (("D", "E"), ("E", "S"), "SAME"),
        ],
    )
    _test_compute_core_point_relation(
        # AAAAAA
        #   BB
        relations=[
            ("A", "B", "INCLUDES"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("A", "E"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # AAAAAA
        # BB
        relations=[
            ("B", "A", "BEGINS"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("B", "E"), ("A", "E"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # AAAAAA
        #     BB
        relations=[
            ("B", "A", "ENDS"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "BEFORE"),
            (("A", "E"), ("B", "E"), "SAME"),
        ],
    )
    _test_compute_core_point_relation(
        # AAAAAA
        # BBCCDD
        relations=[
            ("A", "C", "INCLUDES"),
            ("B", "A", "BEGINS"),
            ("D", "A", "ENDS"),
            ("B", "C", "IBEFORE"),
            ("C", "D", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "E"), ("D", "E"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("C", "E"), ("D", "S"), "SAME"),
        ],
    )
    _test_compute_core_point_relation(
        # AAAAAAA
        # BBCC DD
        relations=[
            ("A", "C", "INCLUDES"),
            ("B", "A", "BEGINS"),
            ("D", "A", "ENDS"),
            ("B", "C", "IBEFORE"),
            ("C", "D", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "E"), ("D", "E"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("C", "E"), ("D", "S"), "BEFORE"),
        ],
    )
    _test_compute_core_point_relation(
        # AABBBCC
        #   DDD
        relations=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
            ("A", "D", "IBEFORE"),
            ("D", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("A", "E"), ("D", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("B", "E"), ("D", "E"), "SAME"),
        ],
    )


def _test_compute_core_point_n_entity_relation(
    relations: List[TimeMlRelation],
    expect_point_relation: List[TmlPointRelation],
    expect_entity_relation: List[TimeMlRelation],
):
    graph = TmlGraph()
    _, _, closure_violation = graph.safe_add_relations(relations)
    assert closure_violation == [], "test must not have violations"

    point_relations = graph.compute_core_point_relations()
    assert sorted(point_relations) == sorted(expect_point_relation)

    entity_relations = graph.compute_core_entity_relations()
    assert sorted(entity_relations) == sorted(expect_entity_relation)


def test_compute_core_point_n_entity_relation():
    """Test computing core point relations"""
    _test_compute_core_point_n_entity_relation(
        # A-->B
        relations=[
            ("A", "B", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "BEFORE"),
        ],
        expect_entity_relation=[
            ("A", "B", "BEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AABB
        relations=[
            ("A", "B", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "B", "IBEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AABBCC
        relations=[
            ("A", "C", "BEFORE"),
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AA
        # BB
        relations=[
            ("A", "B", "IDENTITY"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "E"), ("B", "E"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "B", "IDENTITY"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AA
        # BB
        # CC
        relations=[
            ("A", "B", "IDENTITY"),
            ("B", "C", "IDENTITY"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "S"), ("C", "S"), "SAME"),
            (("A", "E"), ("B", "E"), "SAME"),
            (("A", "E"), ("C", "E"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "B", "IDENTITY"),
            ("A", "C", "IDENTITY"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AA CC
        # BB DD
        relations=[
            ("A", "B", "IDENTITY"),
            ("C", "D", "IDENTITY"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "E"), ("B", "E"), "SAME"),
            (("C", "S"), ("D", "S"), "SAME"),
            (("C", "E"), ("D", "E"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "B", "IDENTITY"),
            ("C", "D", "IDENTITY"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AA
        # BBBB
        relations=[
            ("A", "B", "BEGINS"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "E"), ("B", "E"), "BEFORE"),
        ],
        expect_entity_relation=[
            ("A", "B", "BEGINS"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # XXXXXX
        # AABBCC
        relations=[
            ("A", "C", "BEFORE"),
            ("A", "X", "BEGINS"),
            ("C", "X", "ENDS"),
            ("X", "B", "INCLUDES"),
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("B", "S"), "SAME"),
            (("A", "S"), ("X", "S"), "SAME"),
            (("B", "E"), ("C", "S"), "SAME"),
            (("C", "E"), ("X", "E"), "SAME"),
        ],
        expect_entity_relation=[
            ("A", "X", "BEGINS"),
            ("C", "X", "ENDS"),
            ("A", "B", "IBEFORE"),
            ("B", "C", "IBEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # AAAA
        # BBBBBB
        #   CC
        relations=[
            ("A", "B", "BEGINS"),
            ("B", "C", "INCLUDES"),
            ("C", "A", "ENDS"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "SAME"),
            (("A", "S"), ("C", "S"), "BEFORE"),
            (("A", "E"), ("C", "E"), "SAME"),
            (("A", "E"), ("B", "E"), "BEFORE"),
        ],
        expect_entity_relation=[
            ("A", "B", "BEGINS"),
            ("C", "A", "ENDS"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # DD BBCC
        relations=[
            ("D", "B", "BEFORE"),
            ("D", "C", "BEFORE"),
            ("B", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("D", "E"), ("B", "S"), "BEFORE"),
            (("B", "E"), ("C", "S"), "SAME"),
        ],
        expect_entity_relation=[
            ("D", "B", "BEFORE"),
            ("B", "C", "IBEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation(
        # DD BBCC
        #   AAA
        relations=[
            ("D", "B", "BEFORE"),
            ("D", "C", "BEFORE"),
            ("B", "C", "IBEFORE"),
            ("D", "A", "IBEFORE"),
            ("B", "A", "ENDS"),
            ("A", "C", "IBEFORE"),
        ],
        expect_point_relation=[
            (("A", "S"), ("B", "S"), "BEFORE"),
            (("A", "S"), ("D", "E"), "SAME"),
            (("A", "E"), ("B", "E"), "SAME"),
            (("A", "E"), ("C", "S"), "SAME"),
        ],
        expect_entity_relation=[
            ("D", "A", "IBEFORE"),
            ("A", "C", "IBEFORE"),
            ("B", "A", "ENDS"),
        ],
    )


def _test_compute_core_entity_relation(
    relations: List[TimeMlRelation],
    expect_entity_relation: List[TimeMlRelation],
):
    graph = TmlGraph()
    _, _, closure_violation = graph.safe_add_relations(relations)
    assert closure_violation == [], "test must not have violations"

    entity_relations = graph.compute_core_entity_relations()
    assert len(entity_relations) == len(expect_entity_relation)

    expect_graph = TmlGraph()
    _, _, closure_violation = expect_graph.safe_add_relations(expect_entity_relation)
    assert closure_violation == [], "test must not have violations"
    expect_point_relation = expect_graph.compute_core_point_relations()
    point_relations = graph.compute_core_point_relations()
    assert sorted(point_relations) == sorted(expect_point_relation)


def test_compute_core_enitiy_relation():
    _test_compute_core_entity_relation(
        # DD BBCC
        #   AAA
        # XX YYZZ
        #   MMM
        relations=[
            ("D", "B", "BEFORE"),
            ("D", "C", "BEFORE"),
            ("B", "C", "IBEFORE"),
            ("D", "A", "IBEFORE"),
            ("B", "A", "ENDS"),
            ("A", "C", "IBEFORE"),
            ("A", "M", "IDENTITY"),
            ("D", "X", "IDENTITY"),
            ("B", "Y", "IDENTITY"),
            ("C", "Z", "IDENTITY"),
        ],
        expect_entity_relation=[
            ("D", "A", "IBEFORE"),
            ("A", "C", "IBEFORE"),
            ("B", "A", "ENDS"),
            ("A", "M", "IDENTITY"),
            ("D", "X", "IDENTITY"),
            ("B", "Y", "IDENTITY"),
            ("C", "Z", "IDENTITY"),
        ],
    )
    _test_compute_core_entity_relation(
        # BBBCCC
        #    JJJJJ
        #      DDD
        relations=[
            ("B", "J", "IBEFORE"),
            ("C", "J", "BEGINS"),
            ("D", "J", "ENDS"),
        ],
        expect_entity_relation=[
            ("B", "J", "IBEFORE"),
            ("C", "J", "BEGINS"),
            ("D", "J", "ENDS"),
        ],
    )
    _test_compute_core_entity_relation(
        # AABBBCCCDD
        # IIIIIJJJJJ
        relations=[
            ("A", "I", "BEGINS"),
            ("B", "I", "ENDS"),
            ("C", "J", "BEGINS"),
            ("D", "J", "ENDS"),
            ("I", "J", "IBEFORE"),
        ],
        expect_entity_relation=[
            ("A", "I", "BEGINS"),
            ("B", "I", "ENDS"),
            ("C", "J", "BEGINS"),
            ("D", "J", "ENDS"),
            ("I", "J", "IBEFORE"),
        ],
    )


def _test_check_relation_match_w_violation(
    relations: List[TimeMlRelation],
    expect_match: List[TimeMlRelationWU],
    expect_no_match: List[TimeMlRelationWU],
):
    graph = TmlGraph()
    for relation in relations:
        graph.add_relation(*relation, auto_add_node=True)

    assert graph.has_closure_violation(), "test should have violation"

    for src, tgt, rel in expect_match:
        assert graph.check_relation_match(
            src, tgt, rel
        ), f"{src}, {tgt}, {rel} should match but does not"
    for src, tgt, rel in expect_no_match:
        assert not graph.check_relation_match(
            src, tgt, rel
        ), f"{src}, {tgt}, {rel} should not match but does"


ALL_REL: List[TemporalNormRelTypeWU] = [
    "BEFORE",
    "AFTER",
    "IBEFORE",
    "IAFTER",
    "BEGINS",
    "ENDS",
    "BEGUN_BY",
    "ENDED_BY",
    "INCLUDES",
    "IS_INCLUDED",
    "UNK",
    "IDENTITY",
]


def test_timeml_check_relation_match_w_violation():
    """Test graph ability to perform relation matching"""
    _test_check_relation_match_w_violation(
        relations=[
            # basic violation with two entities
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
        ],
        # allow matching of all relations
        expect_match=[("A", "B", rel) for rel in ALL_REL],
        expect_no_match=[],
    )
    # _test_check_relation_match_w_violation(
    #     relations=[
    #         # violation can be for both BEFORE, AFTER and IDENTITY
    #         ("A", "B", "BEFORE"),
    #         ("A", "B", "IDENTITY"),
    #     ],
    #     expect_match=[
    #         ("A", "B", "BEFORE"),
    #         ("A", "B", "AFTER"),
    #         ("A", "B", "IBEFORE"),
    #         ("A", "B", "IAFTER"),
    #         ("A", "B", "BEGINS"),
    #         ("A", "B", "ENDS"),
    #         ("A", "B", "BEGUN_BY"),
    #         ("A", "B", "ENDED_BY"),
    #         ("A", "B", "INCLUDES"),
    #         ("A", "B", "IS_INCLUDED"),
    #         ("A", "B", "UNK"),
    #         ("A", "B", "IDENTITY"),
    #     ],
    #     expect_no_match=[],
    # )
    _test_check_relation_match_w_violation(
        relations=[
            # violation with IBEFORE only invalidate the start point of B
            ("A", "B", "BEFORE"),
            ("A", "B", "IBEFORE"),
        ],
        expect_match=[
            ("A", "B", "BEFORE"),
            ("A", "B", "IBEFORE"),
            ("A", "B", "UNK"),
        ],
        expect_no_match=[
            ("A", "B", "IDENTITY"),
            ("A", "B", "AFTER"),
            ("A", "B", "IAFTER"),
            ("A", "B", "BEGINS"),
            ("A", "B", "BEGUN_BY"),
            ("A", "B", "ENDS"),
            ("A", "B", "ENDED_BY"),
            ("A", "B", "INCLUDES"),
            ("A", "B", "IS_INCLUDED"),
        ],
    )
    _test_check_relation_match_w_violation(
        # cycle violation
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "C", "BEFORE"),
            ("C", "A", "BEFORE"),
        ],
        expect_match=[
            (src, tgt, rel)
            for src in ["A", "B", "C"]
            for tgt in ["A", "B", "C"]
            if src != tgt
            for rel in ALL_REL
        ],
        expect_no_match=[],
    )
    _test_check_relation_match_w_violation(
        # cycle violation: backward
        relations=[
            ("A", "B", "AFTER"),
            ("B", "C", "AFTER"),
            ("C", "A", "AFTER"),
        ],
        expect_match=[
            (src, tgt, rel)
            for src in ["A", "B", "C"]
            for tgt in ["A", "B", "C"]
            if src != tgt
            for rel in ALL_REL
        ],
        expect_no_match=[],
    )
    _test_check_relation_match_w_violation(
        # violated entities related to valid entity
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "C", "AFTER"),
        ],
        expect_match=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "B", "UNK"),
            ("A", "B", "IDENTITY"),
            ("A", "C", "AFTER"),
            ("B", "C", "AFTER"),
        ],
        expect_no_match=[
            ("A", "C", "BEFORE"),
            ("A", "C", "UNK"),
            ("A", "C", "IDENTITY"),
            ("B", "C", "BEFORE"),
            ("B", "C", "UNK"),
            ("B", "C", "IDENTITY"),
        ],
    )
    _test_check_relation_match_w_violation(
        # violated entities related to valid entity: backward
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "C", "BEFORE"),
        ],
        expect_match=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "B", "UNK"),
            ("A", "B", "IDENTITY"),
            ("A", "C", "BEFORE"),
            ("B", "C", "BEFORE"),
        ],
        expect_no_match=[
            ("A", "C", "AFTER"),
            ("A", "C", "UNK"),
            ("A", "C", "IDENTITY"),
            ("B", "C", "AFTER"),
            ("B", "C", "UNK"),
            ("B", "C", "IDENTITY"),
        ],
    )
    _test_check_relation_match_w_violation(
        # IBEFORE behaves differently, only part of the relation is violated.
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "C", "IBEFORE"),
        ],
        expect_match=[
            ("A", "C", "BEFORE"),
            ("A", "C", "IBEFORE"),
            ("A", "C", "UNK"),
            ("A", "C", "BEGINS"),
            ("A", "C", "IS_INCLUDED"),
            ("B", "C", "BEFORE"),
            ("B", "C", "IBEFORE"),
            ("B", "C", "UNK"),
            ("B", "C", "BEGINS"),
            ("B", "C", "IS_INCLUDED"),
        ]
        + [("A", "B", rel) for rel in ALL_REL],
        expect_no_match=[
            ("A", "C", "IDENTITY"),
            ("A", "C", "AFTER"),
            ("A", "C", "IAFTER"),
            ("A", "C", "BEGUN_BY"),
            ("A", "C", "ENDS"),
            ("A", "C", "ENDED_BY"),
            ("A", "C", "INCLUDES"),
            ("B", "C", "IDENTITY"),
            ("B", "C", "AFTER"),
            ("B", "C", "IAFTER"),
            ("B", "C", "BEGUN_BY"),
            ("B", "C", "ENDS"),
            ("B", "C", "ENDED_BY"),
            ("B", "C", "INCLUDES"),
        ],
    )
    _test_check_relation_match_w_violation(
        relations=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "C", "IAFTER"),
        ],
        expect_match=[
            ("A", "B", "BEFORE"),
            ("A", "B", "AFTER"),
            ("A", "B", "UNK"),
            ("A", "B", "IDENTITY"),
            ("A", "C", "AFTER"),
            ("A", "C", "IAFTER"),
            ("A", "C", "UNK"),
            ("B", "C", "AFTER"),
            ("B", "C", "IAFTER"),
            ("B", "C", "UNK"),
        ],
        expect_no_match=[
            ("A", "C", "BEFORE"),
            ("A", "C", "IDENTITY"),
            ("A", "C", "IBEFORE"),
            ("B", "C", "BEFORE"),
            ("B", "C", "IDENTITY"),
            ("B", "C", "IBEFORE"),
        ],
    )


def _test_compute_core_point_n_entity_relation_w_violation(
    relations: List[TimeMlRelation],
    expect_point_relation: List[TmlPointRelation],
    expect_entity_relation: List[TimeMlRelation],
):
    graph = TmlGraph()
    for relation in relations:
        graph.add_relation(*relation, auto_add_node=True)

    assert graph.has_closure_violation()

    point_relations = graph.compute_core_point_relations()
    assert sorted(point_relations) == sorted(expect_point_relation)

    entity_relations = graph.compute_core_entity_relations()
    assert sorted(entity_relations) == sorted(expect_entity_relation)


def test_compute_core_point_n_entity_relation_w_violation():
    """Test computing core point relations"""
    _test_compute_core_point_n_entity_relation_w_violation(
        # A<-->B-->C
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("A", "C", "BEFORE"),
        ],
        expect_point_relation=[
            (("A", "E"), ("C", "S"), "BEFORE"),
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("A", "C", "BEFORE"),
        ],
    )
    _test_compute_core_point_n_entity_relation_w_violation(
        # C-->|A<-->B
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("C", "A", "IBEFORE"),
        ],
        expect_point_relation=[
            (("C", "S"), ("A", "E"), "BEFORE"),
            (("A", "E"), ("C", "E"), "AMBIGUOUS"),
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("C", "A", "IBCYCLE"),
        ],
    )
    _test_compute_core_point_n_entity_relation_w_violation(
        # C-->|A<-->B
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("C", "A", "IAFTER"),
        ],
        expect_point_relation=[
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "S"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("C", "A", "IACYCLE"),
        ],
    )
    _test_compute_core_point_n_entity_relation_w_violation(
        # |<A---->|<-->B
        # |<C>|
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("C", "A", "BEGINS"),
        ],
        expect_point_relation=[
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "E"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("A", "C", "CYCLE"),
        ],
    )
    _test_compute_core_point_n_entity_relation_w_violation(
        # |<A---->|<-->B
        #     |<C>|
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("C", "A", "ENDS"),
        ],
        expect_point_relation=[
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "E"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("A", "C", "CYCLE"),
        ],
    )
    _test_compute_core_point_n_entity_relation_w_violation(
        # |<A---->|<-->B
        #   |<C>|
        relations=[
            ("A", "B", "BEFORE"),
            ("B", "A", "BEFORE"),
            ("A", "C", "INCLUDES"),
        ],
        expect_point_relation=[
            (("A", "E"), ("A", "S"), "AMBIGUOUS"),
            (("A", "E"), ("B", "E"), "AMBIGUOUS"),
            (("A", "E"), ("B", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "S"), "AMBIGUOUS"),
            (("A", "E"), ("C", "E"), "AMBIGUOUS"),
        ],
        expect_entity_relation=[
            ("A", "B", "CYCLE"),
            ("A", "C", "CYCLE"),
        ],
    )


def test_speed():
    rels = [
        ("ei1000009", "t1007", "BEFORE"),
        ("ei1003", "ei40", "AFTER"),
        ("ei32", "ei36", "BEFORE"),
        ("ei43", "ei49", "BEFORE"),
        ("ei1010", "ei1000012", "BEFORE"),
        ("t2", "t1", "AFTER"),
        ("ei1002", "t0", "BEFORE"),
        ("ei25", "t0", "AFTER"),
        ("ei49", "t10", "BEFORE"),
        ("ei11", "ei13", "BEFORE"),
        ("ei1", "ei2", "BEFORE"),
        ("ei44", "t9", "BEFORE"),
        ("ei2", "t0", "AFTER"),
        ("ei6", "ei7", "BEFORE"),
        ("ei53", "ei49", "BEFORE"),
        ("t1005", "t0", "BEFORE"),
        ("ei3", "ei2", "AFTER"),
        ("ei4", "ei6", "BEFORE"),
        ("ei39", "ei37", "BEFORE"),
        ("ei50", "ei49", "AFTER"),
        ("ei18", "ei16", "BEFORE"),
        ("ei1010", "t5", "AFTER"),
        ("ei49", "t0", "BEFORE"),
        ("ei19", "ei20", "AFTER"),
        ("ei53", "ei1002", "AFTER"),
        ("ei12", "t0", "AFTER"),
        ("ei13", "t0", "BEFORE"),
        ("t10", "t0", "AFTER"),
        ("ei26", "t8", "AFTER"),
        ("ei29", "t0", "BEFORE"),
        ("ei20", "t0", "BEFORE"),
        ("ei1000011", "t1005", "AFTER"),
        ("ei27", "ei26", "AFTER"),
        ("ei10", "ei13", "AFTER"),
        ("ei1000009", "t1", "AFTER"),
        ("ei14", "t1000", "BEFORE"),
        ("ei8", "t0", "BEFORE"),
        ("ei9", "ei10", "BEFORE"),
        ("ei7", "t0", "AFTER"),
        ("ei41", "ei40", "AFTER"),
        ("ei5", "ei6", "BEFORE"),
        ("ei42", "ei40", "AFTER"),
        ("ei23", "ei24", "BEFORE"),
        ("t9", "t0", "AFTER"),
        ("ei51", "ei49", "AFTER"),
        ("ei17", "ei16", "BEFORE"),
        ("ei48", "ei43", "AFTER"),
        ("ei24", "ei25", "BEFORE"),
        ("ei8", "t1", "BEFORE"),
        ("ei21", "ei20", "AFTER"),
        ("ei16", "t0", "AFTER"),
        ("ei40", "t0", "BEFORE"),
        ("ei19", "t7", "AFTER"),
        ("ei14", "t0", "BEFORE"),
        ("ei6", "t0", "BEFORE"),
        ("ei1000008", "t2", "BEFORE"),
        ("ei26", "ei29", "BEFORE"),
        ("ei44", "ei43", "BEFORE"),
        ("ei1000010", "t1", "BEFORE"),
        ("ei43", "t0", "BEFORE"),
        ("ei22", "t0", "BEFORE"),
        ("ei22", "ei24", "BEFORE"),
        ("ei12", "t4", "AFTER"),
        ("ei47", "ei43", "AFTER"),
        ("ei8", "ei10", "BEFORE"),
        ("ei2", "ei6", "BEFORE"),
        ("ei11", "t3", "BEFORE"),
        ("ei12", "ei13", "AFTER"),
        ("ei26", "t0", "BEFORE"),
    ]
    graph = TmlGraph()
    for rel in rels:
        graph.add_relation(*rel, auto_add_node=True)
    graph.has_closure_violation()
    core_ent_rels = graph.compute_core_entity_relations()
