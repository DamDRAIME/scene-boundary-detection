from functools import cache

import torch

from sbd.common.utils.gpu_check import is_cuda_available
from sbd.coregistration.distance import ProximityMetric, compute_pairwise_distance
from sbd.coregistration.dtw.models import Method


def compute_cost_matrix(a: torch.Tensor, b: torch.Tensor, metric: ProximityMetric) -> torch.Tensor:
    """Compute the local cost matrix between two sequences.

    Each cell (i, j) holds the pairwise distance between element i of `a` and element j of `b` under `metric`.

    Args:
        a (torch.Tensor): First sequence of embeddings/features, shape (n, ...).
        b (torch.Tensor): Second sequence of embeddings/features, shape (m, ...).
        metric (ProximityMetric): Distance/similarity metric used to compare elements of `a` and `b`.

    Returns:
        torch.Tensor: Cost matrix of shape (n, m).
    """
    return compute_pairwise_distance(a, b, metric)


def initialize_accumulated_cost_matrix(cost_matrix: torch.Tensor, method: Method) -> torch.Tensor:
    """Initialize the accumulated cost matrix (ACM) boundary conditions for a DTW variant.

    Allocates a zero matrix the same shape as `cost_matrix` and fills column 0 with the cumulative sum of the
    local cost (standard for both variants), and row 0 either as a cumulative sum (Classic, forcing the warping
    path to start at cell (0, 0)) or as a raw copy of the local cost (Subsequence, allowing the path to start at
    any position along the Reference sequence without penalty). The remaining interior cells are left as zero and
    must be filled in by a subsequent ACM computation step (e.g. `compute_accumulated_cost_matrix_cpu`).

    Args:
        cost_matrix (torch.Tensor): Local cost matrix, shape (n, m), as produced by `compute_cost_matrix`.
        method (Method): DTW variant that determines the row-0 boundary condition.

    Raises:
        NotImplementedError: If `method` is not one of the supported `Method` values.

    Returns:
        torch.Tensor: Accumulated cost matrix of shape (n, m) with row 0 and column 0 initialized and all other
            cells set to zero.
    """
    acm = torch.zeros_like(cost_matrix)
    acm[:, 0] = cost_matrix[:, 0].cumsum(dim=0)
    if method == Method.CLASSIC:
        # Classic DTW initialization: the first row is initialized with cumulative costs, which means that the optimal
        # path must start at (0,0).
        acm[0, :] = cost_matrix[0, :].cumsum(dim=0)
    elif method == Method.SUBSEQUENCE:
        # This initialization makes it possible to start at any position of the Reference sequence (R) without
        # accumulating any cost, thus realizing the idea of skipping the beginning of R when being matched to the
        # Query sequence (Q).
        acm[0, :] = cost_matrix[0, :]
    else:
        raise NotImplementedError(f"Unknown DTW method: {method}. Must be 'Classic' or 'Subsequence'.")
    return acm


def compute_accumulated_cost_matrix_cpu(cost_matrix: torch.Tensor, method: Method) -> torch.Tensor:
    """Compute the accumulated cost matrix (ACM) via anti-diagonal-vectorized dynamic programming.

    After the boundary conditions are set by `initialize_accumulated_cost_matrix`, every remaining cell (i, j) is
    filled with `cost_matrix[i, j] + min(acm[i-1, j], acm[i, j-1], acm[i-1, j-1])`. Cells sharing an anti-diagonal
    (i + j = k) only depend on cells from the previous anti-diagonal, so each anti-diagonal is updated in one
    vectorized step, processed in increasing order.

    Args:
        cost_matrix (torch.Tensor): Local cost matrix, shape (n, m), as produced by `compute_cost_matrix`.
        method (Method): DTW variant, forwarded to `initialize_accumulated_cost_matrix`.

    Returns:
        torch.Tensor: Fully populated accumulated cost matrix of shape (n, m).
    """
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
    """Lazily build and cache a `torch.compile`-d version of `compute_accumulated_cost_matrix_gpu`.

    Compilation is deferred to the first call (rather than happening at import time) and the compiled callable is
    memoized via `functools.cache`, so subsequent calls reuse the same compiled kernel instead of recompiling.

    Raises:
        RuntimeError: If no CUDA device with Compute Capability >= 7.0 is available.

    Returns:
        callable: A compiled version of `compute_accumulated_cost_matrix_gpu` with the same signature.
    """
    if not is_cuda_available(min_cc=7):
        msg = "`compute_accumulated_cost_matrix_gpu_compiled` requires CUDA with Compute Capability >= 7.0."
        raise RuntimeError(msg)
    return torch.compile(compute_accumulated_cost_matrix_gpu, backend="inductor")


def _find_optimal_warping_path_subsequence_dtw(
    accumulated_cost_matrix: torch.Tensor, reference_backtrack_idx: int = -1
) -> list[tuple[int, int]]:
    """Find the optimal warping path for Subsequence DTW.

    Starts from the last row of the accumulated cost matrix (the end of the Query sequence, Q) at column
    `reference_backtrack_idx`, and greedily walks to the cheapest of the diagonal, top, and left neighbors at each
    step until row 0 is reached. Unlike Classic DTW, the path is free to start at any column of the Reference
    sequence (R), so backtracking stops as soon as `y == 0` rather than requiring `x == 0` too.

    Args:
        accumulated_cost_matrix (torch.Tensor): Accumulated cost matrix, shape (n, m), as produced by
            `compute_accumulated_cost_matrix_cpu`/`compute_accumulated_cost_matrix_gpu` with `Method.SUBSEQUENCE`.
        reference_backtrack_idx (int, optional): Column of R to start backtracking from. If -1, the optimal column
            is used, i.e. the argmin of the last row of `accumulated_cost_matrix` (ties broken by the earliest/
            smallest index, which yields the shortest possible path). Defaults to -1.

    Returns:
        list[tuple[int, int]]: Sequence of (row/query, column/reference) cell indices forming the optimal warping path,
            ordered from origin (row 0) to destination (last row).
    """
    acm = accumulated_cost_matrix
    y = acm.shape[0] - 1
    # This realizes the idea of skipping the end of R when being matched to the Query sequence.
    x = acm[-1, :].argmin().item() if reference_backtrack_idx == -1 else reference_backtrack_idx
    cell = (y, x)  # Start from destination
    path = [cell]

    while y > 0:
        if x == 0:
            next = (y - 1, 0)
        else:
            _, next = min([acm[i, j], (i, j)] for i, j in [[y - 1, x - 1], [y - 1, x], [y, x - 1]])
        path.append(next)
        y, x = next

    path.reverse()  # From origin to destination
    return path


def _find_optimal_warping_path_classic_dtw(accumulated_cost_matrix: torch.Tensor) -> list[tuple[int, int]]:
    """Find the optimal warping path for Classic DTW.

    Starts from the last cell (bottom-right corner, matching the ends of both sequences) and greedily walks to the
    cheapest of the diagonal, top, and left neighbors at each step until the origin cell (0, 0) is reached, which
    both sequences must start from under this variant.

    Args:
        accumulated_cost_matrix (torch.Tensor): Accumulated cost matrix, shape (n, m), as produced by
            `compute_accumulated_cost_matrix_cpu`/`compute_accumulated_cost_matrix_gpu` with `Method.CLASSIC`.

    Returns:
        list[tuple[int, int]]: Sequence of (row/query, column/reference) cell indices forming the optimal warping path,
            ordered from origin (0, 0) to destination (last row, last column).
    """

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
            _, next = min([acm[i, j], (i, j)] for i, j in [[y - 1, x - 1], [y - 1, x], [y, x - 1]])
        path.append(next)
        cell = next

    path.reverse()  # From origin to destination
    return path


def find_optimal_warping_path(accumulated_cost_matrix: torch.Tensor, method: Method, **kwargs) -> list[tuple[int, int]]:
    """Find the optimal warping path for the given DTW variant.

    Dispatch to the backtracking routine matching the given DTW variant.

    Args:
        accumulated_cost_matrix (torch.Tensor): Accumulated cost matrix to backtrack through, shape (n, m).
        method (Method): DTW variant that determines which backtracking routine is used.
        **kwargs: Extra keyword arguments forwarded to the variant-specific backtracking function (e.g.
            `reference_backtrack_idx` for `Method.SUBSEQUENCE`).

    Raises:
        NotImplementedError: If `method` is not one of the supported `Method` values.

    Returns:
        list[tuple[int, int]]: Sequence of (row/query, column/reference) cell indices forming the optimal warping path,
            ordered from origin to destination.
    """
    if method == Method.CLASSIC:
        return _find_optimal_warping_path_classic_dtw(accumulated_cost_matrix)
    elif method == Method.SUBSEQUENCE:
        return _find_optimal_warping_path_subsequence_dtw(accumulated_cost_matrix, **kwargs)
    else:
        raise NotImplementedError(f"Unknown DTW method: {method}. Must be 'Classic' or 'Subsequence'.")
