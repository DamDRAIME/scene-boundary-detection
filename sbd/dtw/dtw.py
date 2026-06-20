import warnings
from dataclasses import dataclass
from enum import StrEnum, auto

import torch

from sbd.dtw.distance import DistanceFunction, get_distance_fn
from sbd.shared.utils.gpu_check import is_cuda_available


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
    optimal_warping_path: torch.Tensor
    cost_matrix: torch.Tensor | None = None
    accumulated_cost_matrix: torch.Tensor | None = None


def _compute_cost_matrix(a: torch.Tensor, b: torch.Tensor, distance_fn: DistanceFunction) -> torch.Tensor:
    dist_fn = get_distance_fn(distance_fn)
    return dist_fn(a, b)


def _compute_accumulated_cost_matrix_cpu(cost_matrix: torch.Tensor) -> torch.Tensor:
    cm = cost_matrix
    acm = torch.zeros_like(cm)
    acm[0, 0] = cm[0, 0]
    acm[:, 0] = cm[:, 0].cumsum(dim=0)
    acm[0, :] = cm[0, :].cumsum(dim=0)
    # acm[i, j] (on anti-diagonal k=i+j) depends solely on (i-1,j), (i,j-1), (i-1,j-1) —
    # all from the previous anti-diagonal (k-1). Every cell where i+j=k is independent,
    # so the whole strip is one vectorized step. This reduces Python iterations from
    # O(N×M) to O(N+M).
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


def _compute_accumulated_cost_matrix_gpu(cost_matrix: torch.Tensor) -> torch.Tensor:
    """GPU-optimized ACM using a skewed anti-diagonal layout.

    Maps the (n, m) cost matrix to a skewed (n, n+m-1) layout where column k holds all cells on anti-diagonal k=i+j.
    The recurrence in skewed space:

        acm_s[i, k] = cm_s[i, k] + min(
            acm_s[i-1, k-1],   # above  (= acm[i-1, j])
            acm_s[i,   k-1],   # left   (= acm[i,   j-1])
            acm_s[i-1, k-2],   # diag   (= acm[i-1, j-1])
        )

    "Shift down by one row" is implemented as torch.cat([INF, col[:-1]]) so every column update has a constant
    shape (n,).  torch.compile/inductor can therefore fuse the loop into a small fixed set of GPU kernels instead of
    issuing one kernel per Python iteration.
    """
    cm = cost_matrix
    n, m = cm.shape
    total_diags = n + m - 1
    INF = torch.finfo(cm.dtype).max / 2

    # Build coordinate map: j_skewed[i, k] = k - i
    k_range = torch.arange(total_diags, device=cm.device)
    i_range = torch.arange(n, device=cm.device)
    j_skewed = k_range.unsqueeze(0) - i_range.unsqueeze(1)  # (n, total_diags)
    valid = (j_skewed >= 0) & (j_skewed < m)
    i_v, k_v = valid.nonzero(as_tuple=True)
    j_v = j_skewed[i_v, k_v]

    # Skew the cost matrix: invalid cells get INF so they never win the min
    cm_s = torch.full((n, total_diags), INF, dtype=cm.dtype, device=cm.device)
    cm_s[i_v, k_v] = cm[i_v, j_v]

    # ACM buffer with 2 sentinel INF columns prepended so k-1 and k-2 lookups
    # are always in-bounds; buffer column (k+2) stores anti-diagonal k.
    acm_buf = torch.full((n, total_diags + 2), INF, dtype=cm.dtype, device=cm.device)
    acm_buf[:, 2] = cm_s[:, 0]  # anti-diagonal 0: only cell (0,0) is valid

    top = torch.full((1,), INF, dtype=cm.dtype, device=cm.device)

    for k in range(1, total_diags):
        b = k + 2
        prev1 = acm_buf[:, b - 1]  # anti-diagonal k-1
        prev2 = acm_buf[:, b - 2]  # anti-diagonal k-2 (or sentinel)
        above = torch.cat([top, prev1[:-1]])  # acm[i-1, j]
        left = prev1  # acm[i,   j-1]
        diag = torch.cat([top, prev2[:-1]])  # acm[i-1, j-1]
        new_col = cm_s[:, k] + torch.stack([above, left, diag]).min(dim=0).values
        acm_buf[:, b] = torch.where(valid[:, k], new_col, torch.full_like(new_col, INF))

    # De-skew: scatter back to the original (n, m) layout
    acm = torch.empty_like(cm)
    acm[i_v, j_v] = acm_buf[i_v, k_v + 2]
    return acm


# Compiled once; deferred to first call. Only invoked when CC >= 7.0.
if is_cuda_available(min_cc=7):
    _compute_accumulated_cost_matrix_gpu_compiled = torch.compile(
        _compute_accumulated_cost_matrix_gpu, backend="inductor"
    )


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
    use_gpu: bool = False,
) -> DTWOutput:
    assert a.dim() == 2, f"Expected `a` to be a tensor with 2 dimensions, got {a.dim()} dimension(s)."
    assert b.dim() == 2, f"Expected `b` to be a tensor with 2 dimensions, got {b.dim()} dimension(s)."

    cm = _compute_cost_matrix(a, b, distance_fn)

    if use_gpu:
        if not is_cuda_available(min_cc=7):
            msg = "`use_gpu` requires CUDA with Compute Capability >= 7.0. Falling back to the CPU implementation."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            acm = _compute_accumulated_cost_matrix_cpu(cm)
        else:
            acm = _compute_accumulated_cost_matrix_gpu_compiled(cm.cuda()).cpu()
    else:
        acm = _compute_accumulated_cost_matrix_cpu(cm)

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
