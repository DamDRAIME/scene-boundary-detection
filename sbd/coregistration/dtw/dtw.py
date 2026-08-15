import warnings

import torch

from sbd.common.utils.gpu_check import is_cuda_available
from sbd.coregistration.distance import ProximityMetric
from sbd.coregistration.dtw.models import DTWOutput, Method, Window
from sbd.coregistration.dtw.utils import (
    compute_accumulated_cost_matrix_cpu,
    compute_accumulated_cost_matrix_gpu_compiled,
    compute_cost_matrix,
    compute_optimal_warping_path,
)


def dtw(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: ProximityMetric,
    method: Method = Method.CLASSIC,
    # step_pattern,
    # window: Window | None = None,
    keep_artifacts: bool = False,
    use_gpu: bool = False,
) -> DTWOutput:

    assert a.dim() == 2, f"Expected `a` to be a tensor with 2 dimensions, got {a.dim()} dimension(s)."
    assert b.dim() == 2, f"Expected `b` to be a tensor with 2 dimensions, got {b.dim()} dimension(s)."

    cm = compute_cost_matrix(a, b, metric)

    if use_gpu:
        if not is_cuda_available(min_cc=7):
            msg = "`use_gpu` requires CUDA with Compute Capability >= 7.0. Falling back to the CPU implementation."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            acm = compute_accumulated_cost_matrix_cpu(cm, method=method)
        else:
            acm = compute_accumulated_cost_matrix_gpu_compiled(cm.cuda(), method=method).cpu()
    else:
        acm = compute_accumulated_cost_matrix_cpu(cm, method=method)

    path = compute_optimal_warping_path(acm, method=method)
    destination_cell_idx = tuple(path[-1].long().tolist())

    return DTWOutput(
        a,
        b,
        metric,
        method,
        distance=acm[destination_cell_idx].item(),
        optimal_warping_path=path,
        cost_matrix=cm if keep_artifacts else None,
        accumulated_cost_matrix=acm if keep_artifacts else None,
    )
