# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Sequence

import torch

from ..types import BatchType


def infer_batch_size(batch: BatchType, key: Optional[str] = None) -> int:
    if key is not None:
        # use given key for infering the batch size
        return batch[key].shape[0]

    # default case: we always have 'rgb' and/or 'depth' as batched tensor
    # in `batch`
    tensor = batch.get('rgb', batch.get('depth', None))
    return tensor.shape[0]


def move_batch_to_device(
    batch: BatchType,
    device: torch.device,
    keys_to_ignore: Optional[Sequence[str]] = None,
    keys_to_ignore_recursive: Optional[bool] = True
) -> BatchType:
    assert isinstance(batch, dict), "Not implemented"

    for key in list(batch.keys()):
        if keys_to_ignore is not None and key in keys_to_ignore:
            continue

        value = batch[key]
        if isinstance(value, dict):
            # nested dict
            to_ignore = keys_to_ignore if keys_to_ignore_recursive else None
            batch[key] = move_batch_to_device(
                batch=value,
                device=device,
                keys_to_ignore=to_ignore,
            )
        # some tensors (e.g. dense_visual_embedding_lut) cannot be stacked
        # because their sizes differ. for example, each sample in a batch may
        # have a different number of instances, resulting in lookup tables
        # (LUTs) of varying sizes. Since combining LUTs and indices is more
        # efficient when performed on the target device, we iterate over the
        # list and transfer each tensor individually to that device.
        elif isinstance(value, list):
            batch[key] = [
                item.to(device)
                if isinstance(item, torch.Tensor)
                else item  # required to not return empty list for non tensor
                for item in value
            ]
        elif isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    return batch
