import warnings

import torch

from sbd.common.utils.gpu_check import is_cuda_available
from sbd.coregistration.distance import ProximityMetric
from sbd.coregistration.dtw.models import DTWOutput, Method, Window
from sbd.coregistration.dtw.utils import (
    compute_accumulated_cost_matrix_cpu,
    compute_accumulated_cost_matrix_gpu_compiled,
    compute_cost_matrix,
    find_optimal_warping_path,
)


def dtw(
    query: torch.Tensor,
    reference: torch.Tensor,
    metric: ProximityMetric,
    method: Method = Method.CLASSIC,
    # step_pattern,
    # window: Window | None = None,
    keep_artifacts: bool = False,
    use_gpu: bool = False,
) -> DTWOutput:
    """Align two sequences (Reference and Query) using Dynamic Time Warping (DTW).

    Computes the local cost matrix between the two sequences, accumulates it into an ACM (on CPU, or on GPU via a
    compiled kernel when `use_gpu` is set and available), then backtracks the accumulated cost matrix to recover
    the optimal warping path and its total distance.

    Args:
        query (torch.Tensor): Query sequence, shape (n, ...); the sequence being matched.
        reference (torch.Tensor): Reference sequence, shape (m, ...); the sequence being matched against.
        metric (ProximityMetric): Distance/similarity metric used to build the local cost matrix.
        method (Method, optional): DTW variant to use. 2 supported methods are available:
            - `Method.CLASSIC`, both sequences must align start-to-end;
            - `Method.SUBSEQUENCE`, `reference` may be arbitrarily trimmed at both ends to find the best-matching
              subsequence for `query`.
            Defaults to Method.CLASSIC.
        keep_artifacts (bool, optional): If True, retain the intermediate cost matrix and accumulated cost matrix
            on the returned `DTWOutput`; otherwise they are discarded (set to None) to save memory. Defaults to
            False.
        use_gpu (bool, optional): If True, compute the accumulated cost matrix on GPU using a compiled kernel.
            Falls back to the CPU implementation (with a warning) if no CUDA device with Compute Capability >= 7.0
            is available. Defaults to False.

    Returns:
        DTWOutput: The alignment result, including the optimal warping path and its total distance, plus the
            input sequences, metric, method, and (if `keep_artifacts` is True) the cost matrix and accumulated
            cost matrix.
    """

    assert query.dim() == 2, f"Expected `query` to be a tensor with 2 dimensions, got {query.dim()} dimension(s)."
    assert reference.dim() == 2, (
        f"Expected `reference` to be a tensor with 2 dimensions, got {reference.dim()} dimension(s)."
    )

    cm = compute_cost_matrix(query, reference, metric)

    if use_gpu:
        if not is_cuda_available(min_cc=7):
            msg = "`use_gpu` requires CUDA with Compute Capability >= 7.0. Falling back to the CPU implementation."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            acm = compute_accumulated_cost_matrix_cpu(cm, method=method)
        else:
            acm = compute_accumulated_cost_matrix_gpu_compiled(cm.cuda(), method=method).cpu()
    else:
        acm = compute_accumulated_cost_matrix_cpu(cm, method=method)

    path = find_optimal_warping_path(acm, method=method)

    return DTWOutput(
        query,
        reference,
        metric,
        method,
        distance=acm[path[-1]].item(),
        optimal_warping_path=path,
        cost_matrix=cm if keep_artifacts else None,
        accumulated_cost_matrix=acm if keep_artifacts else None,
    )
