from itertools import compress

import torch

from sbd.coregistration.distance import ProximityMetric, compute_pairwise_similarity
from sbd.screenplay.models import SingletonUtterance
from sbd.subtitle.models import SubTitle, Utterance


def get_anchors(
    a: list[SubTitle | Utterance | SingletonUtterance],
    b: list[SubTitle | Utterance | SingletonUtterance],
    metric: ProximityMetric,
    similarity_threshold: float,
    min_count: int = 1,
    max_count: int = 3,
) -> tuple[list[SubTitle | Utterance | SingletonUtterance], list[SubTitle | Utterance | SingletonUtterance]]:
    a_embeddings = torch.stack([torch.tensor(x.embedding) for x in a])
    b_embeddings = torch.stack([torch.tensor(x.embedding) for x in b])
    a_mask, b_mask = _find_high_confidence_matches(
        a_embeddings, b_embeddings, metric, similarity_threshold, min_count=min_count, max_count=max_count
    )
    return list(compress(a, a_mask)), list(compress(b, b_mask))


def _find_high_confidence_matches(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: ProximityMetric,
    similarity_threshold: float,
    min_count: int = 1,
    max_count: int = 3,
) -> tuple[list[bool], list[bool]]:
    similarity = compute_pairwise_similarity(a, b, metric)
    above_threshold = similarity > similarity_threshold
    row_match_count = above_threshold.count_nonzero(dim=1)
    row_mask = (min_count <= row_match_count) & (row_match_count < max_count)
    col_match_count = above_threshold.count_nonzero(dim=0)
    col_mask = (min_count <= col_match_count) & (col_match_count < max_count)
    anchors = above_threshold & row_mask.unsqueeze(1) & col_mask.unsqueeze(0)
    a_anchor_mask = anchors.count_nonzero(dim=1) > 0
    b_anchor_mask = anchors.count_nonzero(dim=0) > 0
    return a_anchor_mask.tolist(), b_anchor_mask.tolist()
