from typing import List, Tuple

from pytest import approx

from ..rep.timeml_graph import TimeMlRelation
from .temporal_awareness import TemporalAwarenessScorer, FMeasure


def _test_temporal_awareness_scorer(
    gold_relations: List[TimeMlRelation],
    pred_relations: List[TimeMlRelation],
    ref_f_measure: FMeasure,
):
    evaluator = TemporalAwarenessScorer()
    evaluator.evaluate_relations(gold_relations, pred_relations)
    f_measure = evaluator.summarize()
    assert f_measure["precision"] == approx(ref_f_measure["precision"], 0.0001)
    assert f_measure["recall"] == approx(ref_f_measure["recall"], 0.0001)
    assert f_measure["fscore"] == approx(ref_f_measure["fscore"], 0.0001)


def test_temporal_awareness_scorer():
    params: List[Tuple[List[TimeMlRelation], List[TimeMlRelation], FMeasure]] = [
        (
            [
                ("t0", "t1", "BEFORE"),
                ("t0", "t2", "BEFORE"),
                ("t1", "t2", "BEFORE"),
                ("t3", "t4", "BEFORE"),
                ("t4", "t5", "BEFORE"),
            ],
            [
                ("t0", "t1", "BEFORE"),
                ("t0", "t2", "BEFORE"),
                ("t1", "t2", "BEFORE"),
                ("t3", "t4", "AFTER"),
                ("t4", "t5", "AFTER"),
            ],
            {
                "precision": 0.6,
                "recall": 0.6,
                "fscore": 0.6,
            },
        ),
        (
            # results dependent on order of relations
            [
                ("t0", "t1", "BEFORE"),
                ("t1", "t2", "BEFORE"),
                ("t0", "t2", "BEFORE"),
                ("t3", "t4", "BEFORE"),
                ("t4", "t5", "BEFORE"),
            ],
            [
                ("t0", "t1", "BEFORE"),
                ("t0", "t2", "BEFORE"),
                ("t1", "t2", "BEFORE"),
                ("t3", "t4", "AFTER"),
                ("t4", "t5", "AFTER"),
            ],
            {
                "precision": 0.6,
                "recall": 0.5,
                "fscore": 0.54545,
            },
        ),
    ]
    for param in params:
        _test_temporal_awareness_scorer(*param)
