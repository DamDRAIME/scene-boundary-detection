from functools import cache

import torch

from sbd.common.utils.gpu_check import is_cuda_available
from sbd.coregistration.distance import ProximityMetric, compute_pairwise_distance
from sbd.coregistration.dtw.models import Method


def compute_cost_matrix(a: torch.Tensor, b: torch.Tensor, metric: ProximityMetric) -> torch.Tensor:
    return compute_pairwise_distance(a, b, metric)


def initialize_accumulated_cost_matrix(cost_matrix: torch.Tensor, method: Method) -> torch.Tensor:
    acm = torch.zeros_like(cost_matrix)
    acm[:, 0] = cost_matrix[:, 0].cumsum(dim=0)
    if method == Method.CLASSIC:
        # Classic DTW initialization: the first row is initialized with cumulative costs, which means that the optimal
        # path must start at (0,0).
        acm[0, :] = cost_matrix[0, :].cumsum(dim=0)
    elif method == Method.SUBSEQUENCE:
        # This initialization makes it possible to start at any position of the sequence Y without accumulating any
        # cost, thus realizing the idea of skipping the beginning of Y when being matched to X.
        acm[0, :] = cost_matrix[0, :]
    else:
        raise NotImplementedError(f"Unknown DTW method: {method}. Must be 'Classic' or 'Subsequence'.")
    return acm


def compute_accumulated_cost_matrix_cpu(cost_matrix: torch.Tensor, method: Method) -> torch.Tensor:
    acm = initialize_accumulated_cost_matrix(cost_matrix, method)
    # acm[i, j] (on anti-diagonal k=i+j) depends solely on (i-1,j), (i,j-1), (i-1,j-1) —
    # all from the previous anti-diagonal (k-1). Every cell where i+j=k is independent,
    # so the whole strip is one vectorized step. This reduces Python iterations from
    # O(N×M) to O(N+M).
    y, x = cost_matrix.shape
    for diag in range(2, y + x - 1):
        i_start = max(1, diag - x + 1)
        i_end = min(y, diag)
        i = torch.arange(i_start, i_end, device=cost_matrix.device)
        j = diag - i
        acm[i, j] = (
            cost_matrix[i, j]
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


def compute_accumulated_cost_matrix_gpu(cost_matrix: torch.Tensor, method: Method) -> torch.Tensor:
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

    Row 0 and column 0 are never explicitly initialized (unlike `compute_accumulated_cost_matrix_cpu`): the "above"
    and "diag" terms are seeded with INF via the `top` sentinel, so row 0 only ever has a "left" neighbor and column
    0 only ever has an "above" neighbor. Running the same min-recurrence there naturally reproduces the classical DTW
    cumsum initialization as a side effect, with no separate init step. For "Subsequence", row 0 additionally drops
    the "left" contribution — matching `initialize_accumulated_cost_matrix`'s `acm[0, :] = cost_matrix[0, :]` — so
    that the warping path can start at any column of `b` without accumulating cost. Column 0 is always cumulative,
    for both types.
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
        if method == Method.SUBSEQUENCE:
            # Row 0: no cumulative "left" contribution, only the local cost — start anywhere for free.
            new_col = torch.cat([cm_s[:1, k], new_col[1:]])
        acm_buf[:, b] = torch.where(valid[:, k], new_col, torch.full_like(new_col, INF))

    # De-skew: scatter back to the original (n, m) layout
    acm = torch.empty_like(cm)
    acm[i_v, j_v] = acm_buf[i_v, k_v + 2]
    return acm


@cache
def compute_accumulated_cost_matrix_gpu_compiled() -> callable:
    # Compiled once; deferred to first call.
    if not is_cuda_available(min_cc=7):
        msg = "`compute_accumulated_cost_matrix_gpu_compiled` requires CUDA with Compute Capability >= 7.0."
        raise RuntimeError(msg)
    return torch.compile(compute_accumulated_cost_matrix_gpu, backend="inductor")


def _compute_optimal_warping_path_subsequence_dtw(
    accumulated_cost_matrix: torch.Tensor, x_idx_start_backtrack: int = -1
) -> torch.Tensor:
    # x_idx_start_backtrack: Index to start back tracking from on X (b*); if set to -1, optimal index is used
    # Choosing the cost-minimizing index in this row (instead of taking the last index as is done in the original DTW
    # approach) realizes the idea of skipping the end of Y when being matched to X.
    # If the optimal index needs to be found and argmin returns multiple indices, the first index is chosen to have
    # the shortest path possible. This is because the first index corresponds to the earliest end of Y.
    acm = accumulated_cost_matrix
    y = acm.shape[0] - 1
    x = acm[-1, :].argmin().item() if x_idx_start_backtrack == -1 else x_idx_start_backtrack
    cell = (y, x)  # Start from destination
    path = [cell]

    while y > 0:
        if x == 0:
            next = (y - 1, 0)
        else:
            _, next = min([acm[i, j], [i, j]] for i, j in [[y - 1, x - 1], [y - 1, x], [y, x - 1]])
        path.append(next)
        y, x = next

    path.reverse()  # From origin to destination
    return torch.Tensor(path)


def _compute_optimal_warping_path_classic_dtw(accumulated_cost_matrix: torch.Tensor) -> torch.Tensor:
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


def compute_optimal_warping_path(accumulated_cost_matrix: torch.Tensor, method: Method, **kwargs) -> torch.Tensor:
    if method == Method.CLASSIC:
        return _compute_optimal_warping_path_classic_dtw(accumulated_cost_matrix)
    elif method == Method.SUBSEQUENCE:
        return _compute_optimal_warping_path_subsequence_dtw(accumulated_cost_matrix, **kwargs)
    else:
        raise NotImplementedError(f"Unknown DTW method: {method}. Must be 'Classic' or 'Subsequence'.")
