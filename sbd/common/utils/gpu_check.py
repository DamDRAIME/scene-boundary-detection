import torch


def is_cuda_available(min_cc: int = 0) -> bool:
    """True when a CUDA device with Compute Capability >= `min_cc` is present."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= min_cc
