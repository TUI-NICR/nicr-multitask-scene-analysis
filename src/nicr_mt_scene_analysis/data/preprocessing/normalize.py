# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np

from ...types import BatchType
from .base import PreprocessingBase


def normalize(
    value: np.array,
    mean: np.array,
    std: np.array,
    dtype: str = 'float32',
    inplace: bool = False
) -> np.array:
    if value.dtype != dtype:
        # convert dtype (if does not match, we have to copy)
        value = value.astype(dtype, copy=True)
    else:
        if not inplace:
            value = value.copy()

    # apply normalization inplace (add spatial axes before)
    value -= mean[np.newaxis, np.newaxis, ...]
    value /= std[np.newaxis, np.newaxis, ...]

    return value


class NormalizeRGB(PreprocessingBase):
    def __init__(
        self,
        output_dtype: str = 'float32',
        multiscale_processing: bool = False,
    ) -> None:
        self._output_dtype = output_dtype

        # prepare normalization parameters
        # RGB: values taken from torchvision (= ImageNet mean and std)
        self._rgb_mean = np.array((0.485, 0.456, 0.406),
                                  dtype=self._output_dtype) * 255
        self._rgb_std = np.array((0.229, 0.224, 0.225),
                                 dtype=self._output_dtype) * 255
        super().__init__(
            fixed_parameters={
                'rgb_mean': self._rgb_mean.tolist(),
                'rgb_std': self._rgb_std.tolist(),
                'output_dtype': self._output_dtype
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        assert sample['rgb'].dtype == 'uint8'

        sample['rgb'] = normalize(sample['rgb'],
                                  mean=self._rgb_mean,
                                  std=self._rgb_std,
                                  dtype=self._output_dtype,
                                  inplace=False)

        return sample, {}


class NormalizeDepth(PreprocessingBase):
    def __init__(
        self,
        depth_mean: float,
        depth_std: float,
        raw_depth: bool = False,
        invalid_depth_value: float = 0.0,
        output_dtype: str = 'float32',
        multiscale_processing: bool = False,
    ) -> None:
        assert depth_std != 0.0

        self._raw_depth = raw_depth
        self._invalid_depth_value = invalid_depth_value
        self._output_dtype = output_dtype

        # prepare normalization parameters
        self._depth_mean = np.array(depth_mean, dtype=self._output_dtype)
        self._depth_std = np.array(depth_std, dtype=self._output_dtype)

        super().__init__(
            fixed_parameters={
                'depth_mean': self._depth_mean.tolist(),
                'depth_std': self._depth_std.tolist(),
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
        if self._raw_depth:
            # get mask of invalid depth values
            invalid_mask = sample['depth'] == self._invalid_depth_value

        sample['depth'] = normalize(sample['depth'],
                                    mean=self._depth_mean,
                                    std=self._depth_std,
                                    dtype=self._output_dtype,
                                    inplace=False)

        if self._raw_depth:
            # reset invalid values (the network should not be able to learn
            # from invalid values)
            sample['depth'][invalid_mask] = self._invalid_depth_value

        return sample, {}
