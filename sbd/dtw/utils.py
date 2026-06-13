from typing import Any, Callable

from torch import Tensor
import torch


def normalize_batch_of_tensors(tensors: Tensor) -> Tensor:
    """
    Normalizes each tensor in `tensors`, so that each one has unit length.

    Args:
        tensors (Tensor): The input tensor batch.

    Returns:
        Tensor: The normalized tensor batch.
    """
    assert tensors.dim() == 2, f"Expected a batch of tensors of shape (Y, X), got {tensors.dim()} dimension(s)."
    if not tensors.is_sparse:
        return torch.nn.functional.normalize(tensors, p=2, dim=1)

    tensors = tensors.coalesce()
    indices, values = tensors.indices(), tensors.values()

    # Compute row norms efficiently
    row_norms = torch.zeros(tensors.size(0), device=tensors.device)
    row_norms.index_add_(0, indices[0], values**2)
    row_norms = torch.sqrt(row_norms).index_select(0, indices[0])

    # Normalize values where norm > 0
    mask = row_norms > 0
    normalized_values = values.clone()
    normalized_values[mask] /= row_norms[mask]

    return torch.sparse_coo_tensor(indices, normalized_values, tensors.size())


def negate_fn(fn: Callable, *args) -> Any:
    return -1 * fn(*args)
