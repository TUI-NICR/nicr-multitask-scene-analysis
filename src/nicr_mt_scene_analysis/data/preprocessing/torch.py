# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ...types import BatchType
from .base import PreprocessingBase
from .utils import _get_relevant_tensor_keys


class ToTorchTensors(PreprocessingBase):
    def __init__(
        self,
        multiscale_processing: bool = True
    ) -> None:
        super().__init__(multiscale_processing=multiscale_processing)

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        for key in _get_relevant_tensor_keys(sample):
            value = sample[key]

            if 3 == value.ndim:
                # (multiple) channel(s) with channels_last: HWC -> CHW
                value = value.transpose((2, 0, 1))
            elif 2 == value.ndim:
                if 'depth' == key:
                    # only one channel without channel axis: HW -> CHW
                    value = value[np.newaxis, ...]
                # otherwise nothing to do, we do not want to add channel axis
                # for masks etc.
            elif 1 == value.ndim:
                if 'dense_visual_embedding_lut' == key:
                    # If the dataset sample has zero instances, the embedding
                    # lut will be empty, resulting in a tensor with ndim==1 and
                    # shape==(0,).
                    # If not explictily handled it only the ndim==2 case (
                    # with instances) will get converted to torch.Tensor
                    # but the ndim==1 case will remain numpy.ndarray.
                    pass
                elif 'image_embedding' == key:
                    # image_embeddings are always ndim==1 and should still
                    # get converted to torch.Tensor.
                    pass
                else:
                    raise ValueError(f"Cannot handle entry '{key}' with "
                                     f"shape '{value.shape}'")
            else:
                raise ValueError(f"Cannot handle entry '{key}' with "
                                 f"shape '{value.shape}'")

            if value.dtype == 'uint16':
                # pytorch does not support uint16
                value = value.astype('int32')

            if value.dtype == 'uint32':
                # pytorch does not support uint32
                value = value.astype('int64')

            # ensure that array is C contiguous
            value = np.ascontiguousarray(value)

            sample[key] = torch.from_numpy(value)

        return sample, {}
