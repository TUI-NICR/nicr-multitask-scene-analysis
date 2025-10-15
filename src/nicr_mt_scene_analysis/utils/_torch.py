# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from functools import wraps

import torch


def _is_mps_available() -> bool:
    # Check if torch has the mps backend attribute
    if not hasattr(torch, 'backends') or not hasattr(torch.backends, 'mps'):
        return False
    return torch.backends.mps.is_available()


_MPS_AVAILABLE = _is_mps_available()


def _is_mps_tensor(x) -> bool:
    return isinstance(x, torch.Tensor) and x.is_mps


def to_cpu_if_mps_tensor(x):
    return x.cpu() if _is_mps_tensor(x) else x


def mps_cpu_fallback(disabled=False):
    # As of Nov 2023: MPS does not support large flattened tensors, e.g., when
    # converting from NxCxHxW to NxCxH*W for a computation. This decorator will
    # move all tensors to CPU to provide a fallback. Tensors are moved:
    # - unless the decorator is disabled
    # - if MPS is available and enabled
    # - regardless of the actual shape (current shape limit seems to 1-65536)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not _MPS_AVAILABLE or disabled:
                # wrapper is disabled -> just call the function
                return f(*args, **kwargs)

            # copy all tensors to CPU
            # TODO (dase6070): apply this recursively (when needed)
            args = [to_cpu_if_mps_tensor(a) for a in args]
            kwargs = {k: to_cpu_if_mps_tensor(v) for k, v in kwargs.items()}

            # apply function
            results = f(*args, **kwargs)

            # TODO (dase6070): copy results back to MPS (when needed)

            return results

        return wrapper

    return decorator


def unit_length(x: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))

    return x / (norm + epsilon)
