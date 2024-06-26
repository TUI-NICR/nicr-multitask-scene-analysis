# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import cv2

from ...types import BatchType
from .utils import _get_input_shape
from .utils import _get_relevant_spatial_keys
from .clone import FlatCloneEntries


# TODO: move fullres stuff out of resize.py
FULLRES_SUFFIX = '_fullres'


def get_fullres_key(key: str) -> str:
    return f'{key}{FULLRES_SUFFIX}'


def get_fullres(sample: BatchType, key: str) -> Any:
    return sample.get(get_fullres_key(key), None)


def get_fullres_shape(sample: BatchType, key: str) -> Tuple[int, int]:
    # try to get fullres shape from given key
    img = get_fullres(sample, key)
    if img is not None:
        return img.shape[-2:]

    # try to get fullres shape from rgb or depth (at least one modality is
    # present)
    img = get_fullres(sample, 'rgb')
    if img is not None:
        return img.shape[-2:]

    img = get_fullres(sample, 'depth')
    if img is not None:
        return img.shape[-2:]

    # we do not know the shape, there was no resizing
    raise ValueError(f"Unable to get fullres shape for `{key}`.")


class FullResCloner(FlatCloneEntries):
    def __init__(
        self,
        keys_to_keep_fullres: Optional[Iterable[str]] = None,
        ignore_missing_keys: bool = True,
    ) -> None:
        super().__init__(
            keys_to_clone=keys_to_keep_fullres,
            ignore_missing_keys=ignore_missing_keys,
            key_prefix='',
            key_suffix=FULLRES_SUFFIX
        )


def resize(
    sample: BatchType,
    height: int,
    width: int,
    keys_to_ignore: Optional[Iterable[str]] = None,
) -> BatchType:
    keys_to_ignore_list = list(keys_to_ignore or [])
    # avoid resizing backups again
    keys_to_ignore_list.extend(
        [k for k in sample if k.endswith(FULLRES_SUFFIX)]
    )

    for key in _get_relevant_spatial_keys(sample, keys_to_ignore_list):
        value = sample[key]

        # determine interpolation
        if 'rgb' == key:
            # color image -> bilinear interpolation
            interpolation = cv2.INTER_LINEAR
        else:
            # for all other entries -> nearest interpolation (depth, masks,
            # segmentation, ...)
            interpolation = cv2.INTER_NEAREST

        # check for boolean masks (OpenCV cannot handle bool inputs)
        fix_bool = value.dtype in (bool, np.bool_)
        if fix_bool:
            value = value.astype('uint8')

        # check for uint32 input (OpenCV cannot handle uint32 inputs, however,
        # we use uint32 for the panoptic key)
        # we circumvent this limitation by viewing the uint32 input as uint8
        # input, i.e, by converting from single-channel uint32 (grayscale
        # uint32) to 4-channel uint8 (rgba uint8)
        # note: this workaround is only possible for nearest interpolation!
        fix_uint32 = False
        if value.dtype == np.uint32 and interpolation == cv2.INTER_NEAREST:
            fix_uint32 = True
            assert value.ndim == 2    # single-channel only
            shape = value.shape
            value = value.view(np.uint8)    # view as 4-channel uint8
            value.shape = (*shape, 4)    # inplace reshape

        # OpenCV does only support some specific dtypes
        assert value.dtype in (
            np.uint8, np.int8, np.uint16, np.int16,
            np.int32, np.float32, np.float64
        )

        # apply resize
        value = cv2.resize(value, (width, height), interpolation=interpolation)

        # bool check part 2
        if fix_bool:
            value = value > 0

        # uint32 check part 2
        if fix_uint32:
            # convert back to single-channel uint32
            value = value.view(np.uint32)[..., 0]

        sample[key] = value

    return sample


class Resize:
    def __init__(
        self,
        height: int,
        width: int,
        keys_to_ignore: Optional[Iterable[str]] = None,
    ) -> None:
        self._height = height
        self._width = width
        self._keys_to_ignore = keys_to_ignore

    def __call__(self, sample: BatchType) -> BatchType:
        return resize(
            sample, self._height, self._width,
            keys_to_ignore=self._keys_to_ignore
        )


class RandomResize:
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
    ) -> None:
        if min_scale < 0 or min_scale > max_scale:
            raise ValueError('Unexpected value of `min_scale`')

        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self, sample: BatchType) -> BatchType:
        # read shape from rgb or depth image (at least one is available)
        h, w = _get_input_shape(sample)

        # get random scale
        if self._min_scale == self._max_scale:
            target_scale = self._min_scale
        else:
            target_scale = np.random.uniform(self._min_scale, self._max_scale)

        # get new height and width
        height = int(round(target_scale*h))
        width = int(round(target_scale*w))

        # resize images in sample
        return resize(sample, height, width)
