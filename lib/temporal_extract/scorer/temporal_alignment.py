from typing import List

from ..rep.timeml_graph import TimeMlRelation, TmlGraph
from .base import FMeasure, RelationScorer, f_measure


class TemporalPointAlignmentScorer(RelationScorer):
    def __init__(self) -> None:
        # counter for precision and recall of the annotaiton and reference
        # recall: found ref in pred
        self.ref_recall = 0
        # total in ref
        self.ref_total = 0
        # precision: found pred in ref
        self.pred_precision = 0
        # total in pred
        self.pred_total = 0

        # count number of documents with closure violation
        self.model_doc_violation_count = 0

        # count number of points in the annotaiton affected by closure violation
        self.pt_affected = 0
        # count total number of points in the annotaiton
        self.pt_total = 0

    def evaluate_relations(
        self, ref_relations: List[TimeMlRelation], pred_relations: List[TimeMlRelation]
    ):
        # evaluation of each document
        doc_ref_recall = 0
        doc_ref_total = 0
        doc_pred_precision = 0
        doc_pred_total = 0

        # create reference graph
        ref_graph = TmlGraph()
        for rel in ref_relations:
            # add each relation to the graph
            ref_graph.add_relation(*rel, auto_add_node=True)
        # NOTE: TE3 does not have closure violations
        assert (
            not ref_graph.has_closure_violation()
        ), "reference must not have violations"

        # compute core point relation of the reference annotation
        ref_core_point_relations = ref_graph.compute_core_point_relations()

        # create prediciton graph
        pred_graph = TmlGraph()
        for rel in pred_relations:
            # add each relation to the graph
            pred_graph.add_relation(*rel, auto_add_node=True)
        if pred_graph.has_closure_violation():
            # track document having closure violation
            self.model_doc_violation_count += 1

        # compute core point relation of the prediction
        pred_core_point_relations = pred_graph.compute_core_point_relations()

        # track total point relations in the prediction
        self.pt_total += len(pred_core_point_relations)

        # keeping only non-violation in prediction graph
        # create an new graph
        pred_graph = TmlGraph()
        for src_pt, tgt_pt, pt_rel in pred_core_point_relations:
            if pt_rel == "AMBIGUOUS":
                # track point relations made ambiguous by the closure violaiton
                self.pt_affected += 1
                # skip point relation which are ambiguous
                continue

            # add non-ambiguous to new graph
            pred_graph.add_point_relation(
                pred_graph.get_point(src_pt, True),
                pred_graph.get_point(tgt_pt, True),
                pt_rel,
            )

        # count prediction recall
        for src_pt, tgt_pt, pt_rel in ref_core_point_relations:
            if src_pt in pred_graph.points and pred_graph.points[
                src_pt
            ].check_point_relation_with(tgt_pt, pt_rel):
                doc_ref_recall += 1
        doc_ref_total = len(ref_core_point_relations)
        self.ref_recall += doc_ref_recall
        self.ref_total += doc_ref_total

        # count prediction precision
        for src_pt, tgt_pt, pt_rel in pred_core_point_relations:
            if src_pt in ref_graph.points and ref_graph.points[
                src_pt
            ].check_point_relation_with(tgt_pt, pt_rel):
                doc_pred_precision += 1
        doc_pred_total = len(pred_core_point_relations)
        self.pred_precision += doc_pred_precision
        self.pred_total += doc_pred_total

        # compute document level evaluation
        doc_r = doc_ref_recall / doc_ref_total
        doc_p = doc_pred_precision / doc_pred_total
        doc_f1 = f_measure(doc_r, doc_p)

        return {
            "pred_precision": doc_pred_precision,
            "pred_total": doc_pred_total,
            "ref_recall": doc_ref_recall,
            "ref_total": doc_ref_total,
            "recall": doc_r,
            "precision": doc_p,
            "f1": doc_f1,
        }

    def summarize(self, beta: float = 1.0) -> FMeasure:
        """computes Temporal Point Alignment.

        Args:
            beta (float, optional): beta for F-score. Defaults to 1.0.

        Returns:
            FMeasure: dict of precision, recall, fscore
        """
        # compute dataset level evaluation
        recall = self.ref_recall / self.ref_total
        precision = self.pred_precision / self.pred_total
        temporal_alignment = f_measure(recall, precision, beta)

        return {
            "precision": precision,
            "recall": recall,
            "fscore": temporal_alignment,
        }


class TemporalEntityAlignmentScorer(RelationScorer):
    def __init__(self) -> None:
        # counter for precision and recall of the annotaiton and reference
        # recall: found ref in pred
        self.ref_recall = 0
        # total in ref
        self.ref_total = 0
        # precision: found pred in ref
        self.pred_precision = 0
        # total in pred
        self.pred_total = 0

        # count number of documents with closure violation
        self.model_doc_violation_count = 0

    def evaluate_relations(
        self, ref_relations: List[TimeMlRelation], pred_relations: List[TimeMlRelation]
    ):
        # evaluation of each document
        doc_ref_recall = 0
        doc_ref_total = 0
        doc_pred_precision = 0
        doc_pred_total = 0

        # create reference graph
        ref_graph = TmlGraph()
        for rel in ref_relations:
            # add each relation to the graph
            ref_graph.add_relation(*rel, auto_add_node=True)
        # NOTE: TE3 does not have closure violations
        assert (
            not ref_graph.has_closure_violation()
        ), "reference must not have violations"

        # compute core entity relation of the reference annotation
        ref_core_ent_relations = ref_graph.compute_core_entity_relations()

        # create prediciton graph
        pred_graph = TmlGraph()
        for rel in pred_relations:
            # add each relation to the graph
            pred_graph.add_relation(*rel, auto_add_node=True)
        if pred_graph.has_closure_violation():
            # track document having closure violation
            self.model_doc_violation_count += 1

        pred_core_ent_relations = pred_graph.compute_core_entity_relations()

        # count prediction recall
        for src, tgt, rel in ref_core_ent_relations:
            if (
                (src, "S") in pred_graph.points
                and (tgt, "S") in pred_graph.points
                and pred_graph.check_relation_match(src, tgt, rel)
            ):
                doc_ref_recall += 1
        doc_ref_total = len(ref_core_ent_relations)
        self.ref_recall += doc_ref_recall
        self.ref_total += doc_ref_total

        # count prediction precision
        for src, tgt, rel in pred_core_ent_relations:
            if (
                (src, "S") in ref_graph.points
                and (tgt, "S") in ref_graph.points
                and ref_graph.check_relation_match(src, tgt, rel)
            ):
                doc_pred_precision += 1
        doc_pred_total = len(pred_core_ent_relations)
        self.pred_precision += doc_pred_precision
        self.pred_total += doc_pred_total

        doc_r = doc_ref_recall / doc_ref_total
        doc_p = doc_pred_precision / doc_pred_total
        doc_f1 = f_measure(doc_r, doc_p)

        return {
            "pred_precision": doc_pred_precision,
            "pred_total": doc_pred_total,
            "ref_recall": doc_ref_recall,
            "ref_total": doc_ref_total,
            "recall": doc_r,
            "precision": doc_p,
            "f1": doc_f1,
        }

    def summarize(self, beta: float = 1.0) -> FMeasure:
        """computes Temporal Entity Alignment.

        Args:
            beta (float, optional): beta for F-score. Defaults to 1.0.

        Returns:
            FMeasure: dict of precision, recall, fscore
        """
        # compute dataset level evaluation
        recall = self.ref_recall / self.ref_total
        precision = self.pred_precision / self.pred_total
        temporal_alignment = f_measure(recall, precision, beta)

        return {
            "precision": precision,
            "recall": recall,
            "fscore": temporal_alignment,
        }
