# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np

from ...types import BatchType
from .base import PreprocessingBase
from .utils import _keys_available


class SemanticClassMapper(PreprocessingBase):
    def __init__(
        self,
        classes_to_map: Tuple[int],
        new_label: int = 0,
        multiscale_processing: bool = True,
        disable_stats: bool = False  # might slightly speed up preprocessing
    ) -> None:
        self._semantic_classes_to_map = np.array(classes_to_map)
        self._new_label = new_label
        self._disable_stats = disable_stats

        super().__init__(
            fixed_parameters={
                'semantic_classes_to_map': self._semantic_classes_to_map,
                'new_label': self._new_label,
                'disable_stats': self._disable_stats
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        if not _keys_available(sample, ('semantic',)):
            # might be an inference call
            return sample, {}

        # get mask for all classes that should be mapped
        mask = np.isin(sample['semantic'],
                       self._semantic_classes_to_map)

        # opt. determine mapping stats
        if not self._disable_stats:
            # add stats to sample
            classes, cnts = np.unique(sample['semantic'][mask],
                                      return_counts=True)
            dynamic_parameters = {'mapped_pixels': dict(zip(classes, cnts))}
        else:
            dynamic_parameters = {}

        # apply mapping inplace
        sample['semantic'][mask] = self._new_label

        return sample, dynamic_parameters
