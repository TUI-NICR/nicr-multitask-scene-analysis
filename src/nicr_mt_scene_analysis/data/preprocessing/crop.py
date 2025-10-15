# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ...types import BatchType
from .base import PreprocessingBase
from .resize import resize
from .utils import _get_input_shape
from .utils import _get_relevant_spatial_keys


class RandomCrop(PreprocessingBase):
    def __init__(
        self,
        crop_height: int,
        crop_width: int,
        keys_to_ignore: Optional[Iterable[str]] = None
    ) -> None:

        self._crop_height = crop_height
        self._crop_width = crop_width
        self._keys_to_ignore = keys_to_ignore
        super().__init__(
            fixed_parameters={
                'crop_height': self._crop_height,
                'crop_width': self._crop_width,
                'keys_to_ignore': self._keys_to_ignore,
            },
            multiscale_processing=False,
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        h, w = _get_input_shape(sample)

        # resize image if it is too small
        scale = 1.0
        if h <= self._crop_height:
            # resize height to match at least crop height
            scale = max(self._crop_height / h, scale)
        if w <= self._crop_width:
            # resize width to match at least crop width
            scale = max(self._crop_width / w, scale)

        if scale > 1.0:
            h, w = int(h*scale+0.5), int(w*scale+0.5)
            sample = resize(sample, height=h, width=w,
                            keys_to_ignore=self._keys_to_ignore)

        # determine slices
        if (h-self._crop_height) > 0:
            y_start = np.random.randint(0, h - self._crop_height)
        else:
            y_start = 0
        if (w-self._crop_width) > 0:
            x_start = np.random.randint(0, w - self._crop_width)
        else:
            x_start = 0
        slice_y = slice(y_start, y_start+self._crop_height)
        slice_x = slice(x_start, x_start+self._crop_width)

        # apply random crop
        for key in _get_relevant_spatial_keys(
            sample, keys_to_ignore=self._keys_to_ignore
        ):
            sample[key] = sample[key][slice_y, slice_x, ...]

        return sample, {'was_resized': scale != 1.0,
                        'resize_height': h,
                        'resize_width': w,
                        'crop_slice_y': slice_y,
                        'crop_slice_x': slice_x}
