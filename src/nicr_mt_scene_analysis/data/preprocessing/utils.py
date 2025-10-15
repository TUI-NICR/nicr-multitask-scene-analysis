# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ...types import BatchType
from .base import PreprocessingBase
from .clone import DEFAULT_CLONE_KEY


class KeyCleaner(PreprocessingBase):
    def __init__(
        self,
        keys_to_clean: Tuple[str],
        multiscale_processing: bool = True,
        **kwargs
    ) -> None:
        self._keys_to_clean = keys_to_clean if keys_to_clean is not None else []
        super().__init__(
            fixed_parameters={
                'keys_to_clean': self._keys_to_clean
            },
            multiscale_processing=multiscale_processing,
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        for key in self._keys_to_clean:
            if key in sample:
                del sample[key]
        return sample, {}


def _keys_available(sample: BatchType, keys: Tuple[str]) -> bool:
    for key in keys:
        if key not in sample:
            return False
    return True


def _get_input_shape(sample: BatchType):
    if 'rgb' in sample:
        h, w, _ = sample['rgb'].shape
    else:
        h, w = sample['depth'].shape

    return h, w


def _get_relevant_tensor_keys(
    sample: BatchType,
    keys_to_ignore: Union[Tuple[str], None] = (DEFAULT_CLONE_KEY, ),
    min_n_dim: Optional[int] = None
) -> BatchType:
    keys = []
    for key, value in sample.items():
        if keys_to_ignore is not None and key in keys_to_ignore:
            # skip specified entries
            continue

        if not isinstance(value, (np.ndarray, torch.Tensor)):
            # skip primitive types such as int, str, or dict
            continue

        if min_n_dim is not None and value.ndim < min_n_dim:
            # skip entries which are below a certain dim.
            # helpful to e.g. only get spatial keys (2D).
            continue

        # we are interested in this key
        keys.append(key)

    return keys


def _get_relevant_spatial_keys(
    sample: BatchType,
    keys_to_ignore: Union[Tuple[str], None] = (DEFAULT_CLONE_KEY, )
) -> BatchType:
    return _get_relevant_tensor_keys(
        sample=sample,
        keys_to_ignore=keys_to_ignore,
        min_n_dim=2
    )
