from enum import StrEnum, auto
from functools import partial
from typing import Callable

import torch

from sbd.dtw.utils import negate_fn, normalize_batch_of_tensors


class DistanceFunction(StrEnum):
    COSINE = auto()
    DOT_PRODUCT = auto()
    EUCLIDEAN = auto()
    MANHATTAN = auto()


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


def get_distance_fn(distance_fn: DistanceFunction) -> Callable:
    if distance_fn == DistanceFunction.COSINE:
        return partial(negate_fn, cosine)
    if distance_fn == DistanceFunction.DOT_PRODUCT:
        return partial(negate_fn, dot_product)
    if distance_fn == DistanceFunction.EUCLIDEAN:
        return euclidean
    if distance_fn == DistanceFunction.MANHATTAN:
        return manhattan
    raise NotImplementedError(f"{distance_fn} has not yet been implemented.")
