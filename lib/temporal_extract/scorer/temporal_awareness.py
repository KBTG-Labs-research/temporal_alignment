from typing import List

from ..rep.timeml_graph import TimeMlRelation, TmlGraph
from .base import FMeasure, RelationScorer, f_measure


class TemporalAwarenessScorer(RelationScorer):
    """[WARNING] `TemporalAwarenessScorer` is for score reproduction only. **DO NOT USE** for acutal evaluation!

    For accurate evaluation of temporal relations between system-prediction and ref-standard please use TemporalPointAlignmentScorer or TemporalEntityAlignmentScorer instead.

    The TemporalAwarenessScorer computes "Temporal Awarness score" accoring to the paper titled "UzZaman, Naushad. Interpreting the temporal aspects of language. University of Rochester, 2012." and the reference implementation from https://github.com/naushadzaman/tempeval3_toolkit.

    This reimplementation also inherit all of the quirks of `tempeval3_toolkit` not discussed in the paper, while NOT inheriting the issues present in the original closure graph implementation.

    quirks include
     * greedy removal of closure violation
     * matching relations with removed violation causing relations

    This leads to this implementation of TemporalAwareness favoring outputs with closure violation, which is not ideal for evaluation.
    """

    def __init__(self, suppress_warning: bool = False) -> None:
        # print warning
        if not suppress_warning:
            print(self.__doc__)
        # self.ref_rels: int = 0
        # self.ref_redundant_rels: int = 0
        self.ref_non_redundant_rels: int = 0
        self.ref_closure_violation_rels: int = 0
        self.ref_nr_match: int = 0
        self.ref_cv_match: int = 0

        # self.pred_rels: int = 0
        # self.pred_redundant_rels: int = 0
        self.pred_non_redundant_rels: int = 0
        self.pred_closure_violation_rels: int = 0
        self.pred_nr_match: int = 0
        self.pred_cv_match: int = 0

    def evaluate_relations(
        self, ref_relations: List[TimeMlRelation], pred_relations: List[TimeMlRelation]
    ):
        """Evaluate relation of one TimeML document"""
        ref_graph = TmlGraph()
        ref_nr_relations, ref_r_rels, ref_cv_rels = ref_graph.safe_add_relations(
            ref_relations
        )

        pred_graph = TmlGraph()
        pred_nr_relations, pred_r_rels, pred_cv_rels = pred_graph.safe_add_relations(
            pred_relations
        )

        ref_nr_match_pred = 0
        for rel in ref_nr_relations:
            ref_nr_match_pred += pred_graph.check_relation_match(*rel)
        ref_cv_match_pred = 0
        for rel in ref_cv_rels:
            ref_cv_match_pred += pred_graph.check_relation_match(*rel)

        pred_nr_match_ref = 0
        for rel in pred_nr_relations:
            pred_nr_match_ref += ref_graph.check_relation_match(*rel)
        pred_cv_match_ref = 0
        for rel in pred_cv_rels:
            pred_cv_match_ref += ref_graph.check_relation_match(*rel)

        # self.ref_rels += len(ref_relations)
        # self.ref_redundant_rels += len(ref_r_rels)
        self.ref_non_redundant_rels += len(ref_nr_relations)
        self.ref_closure_violation_rels += len(ref_cv_rels)
        self.ref_nr_match += ref_nr_match_pred
        self.ref_cv_match += ref_cv_match_pred

        # self.pred_rels += len(pred_relations)
        # self.pred_redundant_rels += len(pred_r_rels)
        self.pred_non_redundant_rels += len(pred_nr_relations)
        self.pred_closure_violation_rels += len(pred_cv_rels)
        self.pred_nr_match += pred_nr_match_ref
        self.pred_cv_match += pred_cv_match_ref

        precision = (pred_nr_match_ref + pred_cv_match_ref) / (
            len(pred_nr_relations) + len(pred_cv_rels)
        )
        recall = (ref_nr_match_pred + ref_cv_match_pred) / (
            len(ref_nr_relations) + len(ref_cv_rels)
        )
        fscore = f_measure(precision=precision, recall=recall)

        return {
            "ref_nr_match_pred": ref_nr_match_pred,
            "ref_cv_match_pred": ref_cv_match_pred,
            "ref_nr": len(ref_nr_relations),
            "ref_cv": len(ref_cv_rels),
            "ref": len(ref_nr_relations) + len(ref_cv_rels),
            "pred_nr_match_ref": pred_nr_match_ref,
            "pred_cv_match_ref": pred_cv_match_ref,
            "pred_nr": len(pred_nr_relations),
            "pred_cv": len(pred_cv_rels),
            "pred": len(pred_nr_relations) + len(pred_cv_rels),
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    def summarize(self, beta: float = 1.0) -> FMeasure:
        """computes Temporal Awarness score, see `TemporalAwarenessScorer` class pydoc for more details.

        Args:
            beta (float, optional): beta for F-score. Defaults to 1.0.

        Returns:
            FMeasure: dict of precision, recall, fscore
        """
        precision = (self.pred_nr_match + self.pred_cv_match) / (
            (self.pred_non_redundant_rels + self.pred_closure_violation_rels)
        )
        recall = (self.ref_nr_match + self.ref_cv_match) / (
            self.ref_non_redundant_rels + self.ref_closure_violation_rels
        )
        fscore = f_measure(precision=precision, recall=recall, beta=beta)
        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }
