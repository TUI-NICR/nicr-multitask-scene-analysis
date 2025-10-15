# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import PostprocessingBase


class DensePostprocessingBase(PostprocessingBase):
    def _crop_to_valid_region_and_resize_prediction(
        self,
        prediction: torch.Tensor,
        valid_region_slices: Tuple[slice, slice],
        shape: Tuple[int, int],   # (h, w)
        mode: str = 'nearest'
    ) -> torch.Tensor:
        # crop to valid region (assume ...xHxW)
        slice_h, slice_w = valid_region_slices
        prediction = prediction[..., slice_h, slice_w]

        # check if resize is required
        h, w = shape
        if (h, w) == tuple(prediction.shape[-2:]):
            # nothing to do
            return prediction

        # torch expects BCHW for interpolate (add channel axis if required)
        ndim = prediction.ndim
        if 3 == ndim:
            prediction = prediction.unsqueeze(1)

        # as of pytorch 1.11 only float is supported (except uint8+nearest)
        dtype = prediction.dtype
        if not prediction.is_floating_point():
            prediction = prediction.to(torch.float32)

        # resize
        kwargs = {'align_corners': False} if mode != 'nearest' else {}
        prediction = F.interpolate(
            prediction,
            size=(h, w),
            mode=mode,
            **kwargs
        )

        # restore dtype
        prediction = prediction.to(dtype)

        # restore shape (remove channel dimension if it was added)
        if 3 == ndim:
            prediction = prediction.squeeze(1)

        return prediction
