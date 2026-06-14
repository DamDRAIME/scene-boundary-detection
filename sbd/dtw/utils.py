from typing import Any, Callable

from torch import Tensor
import torch
import torchvision.transforms.functional as F


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


def gaussian_blur(tensors: Tensor) -> Tensor:
    i = 0
    while tensors.dim < 4:
        tensors = tensors.unsqueeze(0)
        i += 1

    t_blurred = F.gaussian_blur(tensors, kernel_size=[5, 5], sigma=[1.0, 1.0])

    while i > 0:
        t_blurred = t_blurred.squeeze()

    return t_blurred
