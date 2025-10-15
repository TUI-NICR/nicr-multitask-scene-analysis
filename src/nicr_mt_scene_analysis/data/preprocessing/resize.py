# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import cv2

from ...types import BatchType
from .base import get_applied_preprocessing_meta
from .base import PreprocessingBase
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


def get_valid_region_slices(sample: BatchType) -> Tuple[slice, slice]:
    # get preprocessing meta
    meta = get_applied_preprocessing_meta(sample)

    # search for Resize preprocessor
    # note, all samples share the same original resolution, so using the
    # first element in batch is fine
    # TODO: change this?
    resize_meta = None
    for pre in meta[0]:
        if Resize.__name__ == pre['type']:
            resize_meta = pre
            break

    if resize_meta is not None:
        return (
            resize_meta['valid_region_slice_y'],
            resize_meta['valid_region_slice_x']
        )

    # we do not know the slices
    raise ValueError("Unable to get get valid region slices.")


def get_valid_region_slices_and_fullres_shape(
    sample: BatchType,
    key: str
) -> Tuple[Tuple[slice, slice], Tuple[int, int]]:
    return get_valid_region_slices(sample), get_fullres_shape(sample, key)


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

    for key in _get_relevant_spatial_keys(
        sample,
        keys_to_ignore=keys_to_ignore_list
    ):
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
        # we circumvent the mentioned limitation by viewing the uint32 input as
        # uint8 input, i.e, by converting from single-channel uint32 (grayscale
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


def pad(
    sample: BatchType,
    padding_top: int,
    padding_bottom: int,
    padding_left: int,
    padding_right: int,
    padding_mode: str = 'zero',
    keys_to_ignore: Optional[Iterable[str]] = None,
) -> BatchType:
    keys_to_ignore_list = list(keys_to_ignore or [])
    # avoid padding backups
    keys_to_ignore_list.extend(
        [k for k in sample if k.endswith(FULLRES_SUFFIX)]
    )

    kwargs_lookup = {
        'zero': {'mode': 'constant', 'constant_values': 0},
        'reflect': {'mode': 'reflect'},
    }

    for key in _get_relevant_spatial_keys(sample, keys_to_ignore_list):
        value = sample[key]

        assert value.ndim in (2, 3)  # with channels last!

        # apply padding
        padding = ((padding_top, padding_bottom),
                   (padding_left, padding_right))
        if value.ndim == 3:
            padding = (*padding, (0, 0))

        value = np.pad(value, padding, **kwargs_lookup[padding_mode])

        sample[key] = value

    return sample


class Resize(PreprocessingBase):
    def __init__(
        self,
        height: int,
        width: int,
        keys_to_ignore: Optional[Iterable[str]] = None,
        keep_aspect_ratio: bool = False,
        padding_mode: str = 'zero',
    ) -> None:
        self._height = height
        self._width = width
        self._keys_to_ignore = keys_to_ignore
        self._keep_aspect_ratio = keep_aspect_ratio
        assert padding_mode in ('zero', 'reflect')
        self._padding_mode = padding_mode

        super().__init__(
            fixed_parameters={
                'keys_to_ignore': self._keys_to_ignore,
                'keep_aspect_ratio': keep_aspect_ratio,
                'padding_mode': padding_mode
            },
            multiscale_processing=False
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        # read shape from rgb or depth image (at least one is available)
        orig_height, orig_width = _get_input_shape(sample)

        if not self._keep_aspect_ratio:
            # resize to fixed size, might distort aspect ratio
            height = self._height
            width = self._width

            # determine padding -> no padding
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

            valid_region_slice_y = slice(0, height)
            valid_region_slice_x = slice(0, width)
        else:
            # resize to fixed size, while keeping aspect ratio
            # determine scale and new height and width
            scale = min(self._height/orig_height, self._width/orig_width)
            height = int(round(scale*orig_height))
            width = int(round(scale*orig_width))
            # determine padding
            pad_height = self._height - height
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_width = self._width - width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            valid_region_slice_y = slice(pad_top, pad_top+height)
            valid_region_slice_x = slice(pad_left, pad_left+width)

        # apply resize
        sample_resized = resize(
            sample,
            height=height, width=width,
            keys_to_ignore=self._keys_to_ignore
        )

        # apply padding
        sample_resized_padded = pad(
            sample_resized,
            padding_top=pad_top, padding_bottom=pad_bottom,
            padding_left=pad_left, padding_right=pad_right,
            padding_mode=self._padding_mode,
            keys_to_ignore=self._keys_to_ignore
        )

        return sample_resized_padded, {
            'old_height': orig_height,
            'old_width': orig_width,
            'new_height': self._height,
            'new_width': self._width,
            'valid_region_slice_y': valid_region_slice_y,
            'valid_region_slice_x': valid_region_slice_x,
        }


class RandomResize(PreprocessingBase):
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        keys_to_ignore: Optional[Iterable[str]] = None,
    ) -> None:
        if min_scale < 0 or min_scale > max_scale:
            raise ValueError('Unexpected value for `min_scale`')

        self._min_scale = min_scale
        self._max_scale = max_scale
        self._keys_to_ignore = keys_to_ignore

        super().__init__(
            fixed_parameters={
                'min_scale': self._min_scale,
                'max_scale': self._max_scale,
                'keys_to_ignore': self._keys_to_ignore
            },
            multiscale_processing=False
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
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
        return resize(
            sample, height, width,
            keys_to_ignore=self._keys_to_ignore
        ), {
            'old_height': h,
            'old_width': w,
            'new_height': height,
            'new_width': width,
            'valid_region_slice_y': slice(0, height),
            'valid_region_slice_x': slice(0, width),
        }
