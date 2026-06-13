from dataclasses import dataclass
from enum import StrEnum, auto

import torch

from sbd.dtw.distance import DistanceFunction, get_distance_fn


@dataclass
class Cell:
    y: int
    x: int

    @property
    def is_origin(self) -> bool:
        return not (self.x or self.y)


class Plot(StrEnum):
    ALIGNMENT = auto()
    CONTOUR = auto()
    TWO_WAY = auto()
    THREE_WAY = auto()


class Window(StrEnum):
    ITAKURA = auto()
    SAKOECHIBA = auto()
    SLANTED_BAND = auto()


@dataclass
class DTWOutput:
    a: torch.Tensor
    b: torch.Tensor
    distance_fn: DistanceFunction
    distance: float
    # distance_normalized: float
    optimal_warping_path: torch.Tensor
    cost_matrix: torch.Tensor | None = None
    accumulated_cost_matrix: torch.Tensor | None = None


def _compute_cost_matrix(a: torch.Tensor, b: torch.Tensor, distance_fn: DistanceFunction) -> torch.Tensor:
    dist_fn = get_distance_fn(distance_fn)
    return dist_fn(a, b)


def _compute_accumulated_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
    cm = cost_matrix
    acm = torch.zeros_like(cm)
    acm[0, 0] = cm[0, 0]
    acm[:, 0] = cm[:, 0].cumsum(dim=0)
    acm[0, :] = cm[0, :].cumsum(dim=0)
    # acm[i, j] (on the anti-diagonal / k=i+j) depends solely on (i-1, j), (i, j-1), and (i-1, j-1) — all from the
    # previous anti-diagonal (i.e.: k-1).
    # So every cell where i+j = k is independent of its neighbors on the same anti-diagonal and can be computed in
    # one vectorized step. The outer loop walks N+M-3 anti-diagonals instead of N×M individual cells.
    y, x = cm.shape
    for diag in range(2, y + x - 1):
        i_start = max(1, diag - x + 1)
        i_end = min(y, diag)
        i = torch.arange(i_start, i_end, device=cm.device)
        j = diag - i
        acm[i, j] = (
            cm[i, j]
            + torch.stack(
                [
                    acm[i - 1, j],  # from above
                    acm[i, j - 1],  # from left
                    acm[i - 1, j - 1],  # from diagonal (behind)
                ]
            )
            .min(dim=0)
            .values
        )
    return acm


def _compute_optimal_warping_path(accumulated_cost_matrix: torch.Tensor) -> torch.Tensor:
    def is_origin(y: int, x: int) -> bool:
        return not (y or x)

    acm = accumulated_cost_matrix
    y, x = acm.shape
    cell = (y - 1, x - 1)  # Start from destination
    path = [cell]

    while not is_origin(*cell):
        y, x = cell
        if y == 0:
            next = (0, x - 1)
        elif x == 0:
            next = (y - 1, 0)
        else:
            _, next = min([acm[i, j], [i, j]] for i, j in [[y - 1, x - 1], [y - 1, x], [y, x - 1]])
        path.append(next)
        cell = next

    path.reverse()  # From origin to destination
    return torch.Tensor(path)


def dtw(
    a: torch.Tensor,
    b: torch.Tensor,
    distance_fn: DistanceFunction,
    # step_pattern,
    # window: Window | None = None,
    keep_artifacts: bool = False,
) -> DTWOutput:
    assert a.dim() == 2, f"Expected `a` to be a tensor with 2 dimensions, got {a.dim()} dimension(s)."
    assert b.dim() == 2, f"Expected `b` to be a tensor with 2 dimensions, got {b.dim()} dimension(s)."

    cm = _compute_cost_matrix(a, b, distance_fn)
    acm = _compute_accumulated_cost_matrix(cm)
    path = _compute_optimal_warping_path(acm)

    return DTWOutput(
        a,
        b,
        distance_fn,
        distance=acm[-1, -1].item(),
        optimal_warping_path=path,
        cost_matrix=cm if keep_artifacts else None,
        accumulated_cost_matrix=acm if keep_artifacts else None,
    )
