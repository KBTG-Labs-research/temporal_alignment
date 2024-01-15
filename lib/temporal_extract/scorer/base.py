from abc import abstractmethod
from typing import Dict, List, TypedDict, Union

from ..rep.timeml_graph import TimeMlRelation


def f_measure(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_sq = beta * beta
    return (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if precision + recall != 0.0
        else 0.0
    )


class FMeasure(TypedDict):
    precision: float
    recall: float
    fscore: float


class RelationScorer:
    @abstractmethod
    def evaluate_relations(
        self, ref_relations: List[TimeMlRelation], pred_relations: List[TimeMlRelation]
    ) -> Dict[str, Union[int, float]]:
        ...

    @abstractmethod
    def summarize(self, beta: float = 1.0) -> FMeasure:
        ...
