# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ...types import BatchType
from .base import PreprocessingBase
from .utils import _get_relevant_spatial_keys


class RandomHorizontalFlip(PreprocessingBase):
    def __init__(
        self,
        p: float,
        keys_to_ignore: Optional[Iterable[str]] = None
    ) -> None:
        self._p = p
        self._keys_to_ignore = keys_to_ignore

        super().__init__(
            fixed_parameters={
                'p': self._p
            },
            multiscale_processing=False
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        # catch common failure case
        if 'orientations_present' in sample:
            raise RuntimeError("Do not apply `RandomHorizontalFlip` after "
                               "`OrientationTargetGenerator`.")

        do_flip = np.random.uniform() <= self._p
        if do_flip:
            # spatial entries
            for key in _get_relevant_spatial_keys(
                sample,
                keys_to_ignore=self._keys_to_ignore
            ):
                sample[key] = np.flip(sample[key], axis=1)

            if 'orientations' in sample:
                # mirror orientations at y axis
                for id_ in sample['orientations']:
                    sample['orientations'][id_] = \
                        (2*np.pi - sample['orientations'][id_]) % (2*np.pi)

        return sample, {'was_flipped': do_flip}
