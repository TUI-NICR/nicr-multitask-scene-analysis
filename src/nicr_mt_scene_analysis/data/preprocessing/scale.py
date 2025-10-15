# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

from ...types import BatchType
from .base import PreprocessingBase


class ScaleDepth(PreprocessingBase):
    def __init__(
        self,
        new_min: float = 0.0,
        new_max: float = 1.0,
        raw_depth: bool = False,
        invalid_depth_value: float = 0.0,
        output_dtype: str = 'float32',
        multiscale_processing: bool = False
    ) -> None:
        self._new_min = new_min
        self._new_max = new_max
        self._raw_depth = raw_depth
        self._invalid_depth_value = invalid_depth_value
        self._output_dtype = output_dtype

        super().__init__(
            fixed_parameters={
                'new_min': self._new_min,
                'new_max': self._new_max,
                'raw_depth': self._raw_depth,
                'invalid_depth_value': self._invalid_depth_value,
                'output_dtype': self._output_dtype
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        depth = sample['depth']  # shorter alias

        if depth.dtype != self._output_dtype:
            # convert dtype (if does not match, we have to copy)
            depth = depth.astype(self._output_dtype, copy=True)

        if self._raw_depth:
            # get mask of invalid depth values
            invalid_mask = depth == self._invalid_depth_value

        # scale to unit min/max and, subsequently, to new min/max
        cur_min, cur_max = depth.min(), depth.max()
        depth = (depth - cur_min) / (cur_max - cur_min)
        depth = depth * (self._new_max - self._new_min) + self._new_min

        if self._raw_depth:
            # reset invalid values (the network should not be able to learn
            # from invalid values)
            depth[invalid_mask] = self._invalid_depth_value

        # assign new depth image
        sample['depth'] = depth

        return sample, {}
