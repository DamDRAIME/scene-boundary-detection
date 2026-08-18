from enum import StrEnum, auto
from functools import partial
from typing import Callable

import torch

from sbd.coregistration.utils import negate_fn, normalize_batch_of_tensors


class ProximityMetric(StrEnum):
    COSINE = auto()
    DOT_PRODUCT = auto()
    EUCLIDEAN = auto()
    MANHATTAN = auto()


def compute_pairwise_distance(a: torch.Tensor, b: torch.Tensor, metric: ProximityMetric) -> torch.Tensor:
    dist_metric = get_distance_fn(metric)
    return dist_metric(a, b)


def compute_pairwise_similarity(a: torch.Tensor, b: torch.Tensor, metric: ProximityMetric) -> torch.Tensor:
    dist_metric = get_similarity_fn(metric)
    return dist_metric(a, b)


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot-product between two tensors, i.e. dot_prod(a[i], b[j]) for all i and j.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = dot_prod(a[i], b[j])
    """
    return torch.mm(a, b.transpose(0, 1))


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors, i.e. cos_sim(a[i], b[j]) for all i and j.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a_norm = normalize_batch_of_tensors(a)
    b_norm = normalize_batch_of_tensors(b)
    return dot_product(a_norm, b_norm)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine distance between two tensors, i.e. 1 - cos_sim(a[i], b[j]) for all i and j.

    Unlike `-cos_sim` (what `negate_fn(cosine, ...)` gives you), this is non-negative (range [0, 2]).
    DTW's accumulated-cost recurrence takes a *minimum sum* of local costs along a path; if the local
    cost is negative on average (as `-cos_sim` is here, since embeddings from most models have a
    positive baseline similarity even for unrelated inputs), summing more terms only drives the total
    further down, so the "cheapest" path degenerates into the *longest possible* one instead of the
    best-aligned one. A non-negative cost keeps every additional cell in the path costing something,
    so DTW is actually incentivized toward short, well-matched paths.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = 1 - cos_sim(a[i], b[j])
    """
    return 1 - cosine(a, b)


def dot_product_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes 1 - dot_product(a[i], b[j]) for all i and j.

    Same rationale as `cosine_distance`: `-dot_product` is unbounded and typically negative on
    average, which biases DTW toward the longest possible path instead of the best-aligned one.

    Unlike `cosine`, `dot_product` isn't intrinsically bounded to [-1, 1] — that only holds when `a`
    and `b` are unit-normalized, in which case `dot_product` *is* `cosine` and this is equivalent to
    `cosine_distance`. That's true for every embedding source currently used in this project (e.g.
    `SentenceTransformer("all-MiniLM-L6-v2").encode(...)` returns unit vectors), but isn't guaranteed
    in general: feeding this un-normalized vectors can reproduce the same unbounded-cost bug.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = 1 - dot_prod(a[i], b[j])
    """
    return 1 - dot_product(a, b)


def manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the manhattan distance between two tensors, i.e. -manhattan_distance(a[i], b[j]) for all i and j.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = manhattan_distance(a[i], b[j])
    """
    return torch.cdist(a, b, p=1.0)


def euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the euclidean distance between two tensors, i.e. euclidean_distance(a[i], b[j]) for all i and j.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: Matrix with res[i][j] = euclidean_distance(a[i], b[j])
    """
    return torch.cdist(a, b, p=2.0)


def get_distance_fn(metric: ProximityMetric) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if metric == ProximityMetric.COSINE:
        return cosine_distance
    if metric == ProximityMetric.DOT_PRODUCT:
        return dot_product_distance
    if metric == ProximityMetric.EUCLIDEAN:
        return euclidean
    if metric == ProximityMetric.MANHATTAN:
        return manhattan
    raise NotImplementedError(f"{metric} has not yet been implemented.")


def get_similarity_fn(metric: ProximityMetric) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if metric == ProximityMetric.COSINE:
        return cosine
    if metric == ProximityMetric.DOT_PRODUCT:
        return dot_product
    if metric == ProximityMetric.EUCLIDEAN:
        return partial(negate_fn, euclidean)
    if metric == ProximityMetric.MANHATTAN:
        return partial(negate_fn, manhattan)
    raise NotImplementedError(f"{metric} has not yet been implemented.")
