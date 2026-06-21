# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ...types import BatchType
from .base import PreprocessingBase


def _stack_masks(
    masks: Sequence[np.ndarray],
    height: int,
    width: int
) -> np.ndarray:
    if not masks:
        # empty case
        return np.zeros((height, width, 0), dtype=bool)
    stacked = np.stack(masks, axis=-1)
    return stacked


def _target_to_array(labels: Sequence[int]) -> np.ndarray:
    if not labels:
        # empty case
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(labels, dtype=np.int64)


class SemanticTokenMaskTargetGenerator(PreprocessingBase):
    def __init__(
        self,
        *,
        semantic_key: str = 'semantic',
        masks_key: str = 'semantic_token_masks',
        labels_key: str = 'semantic_token_labels',
        ignore_labels: Tuple[int, ...] = (0,),
        multiscale_processing: bool = False
    ):
        self._semantic_key = semantic_key
        self._masks_key = masks_key
        self._labels_key = labels_key
        self._ignore_labels = tuple(ignore_labels)
        super().__init__(
            fixed_parameters={
                'semantic_key': self._semantic_key,
                'masks_key': self._masks_key,
                'labels_key': self._labels_key,
                'ignore_labels': self._ignore_labels
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        semantic = sample[self._semantic_key]
        masks = []
        labels = []
        for sem_value in np.unique(semantic):
            sem_label = int(sem_value)
            # skip void for example
            if sem_label in self._ignore_labels:
                continue
            mask = semantic == sem_label
            masks.append(mask.astype(bool))
            labels.append(sem_label)

        height, width = semantic.shape
        sample[self._masks_key] = _stack_masks(masks, height, width)
        sample[self._labels_key] = _target_to_array(labels)

        return sample, {}


class PanopticTokenMaskTargetGenerator(PreprocessingBase):
    def __init__(
        self,
        *,
        panoptic_key: str = 'panoptic',
        masks_key: str = 'panoptic_token_masks',
        labels_key: str = 'panoptic_token_labels',
        ids_key: str = 'panoptic_token_ids',
        void_label: int = 0,
        max_instances_per_category: int = 1 << 16,
        multiscale_processing: bool = False
    ):
        self._panoptic_key = panoptic_key
        self._masks_key = masks_key
        self._labels_key = labels_key
        self._ids_key = ids_key
        self._void_label = void_label
        self._max_instances_per_category = max_instances_per_category
        super().__init__(
            fixed_parameters={
                'panoptic_key': self._panoptic_key,
                'masks_key': self._masks_key,
                'labels_key': self._labels_key,
                'ids_key': self._ids_key,
                'void_label': self._void_label,
                'max_instances_per_category': self._max_instances_per_category
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        panoptic = sample[self._panoptic_key]
        masks = []
        labels = []
        ids = []
        for pann_value in np.unique(panoptic):
            if pann_value == self._void_label:
                continue
            mask = panoptic == pann_value
            if not mask.any():
                continue
            # compute semantic label from panoptic value
            label = int(pann_value // self._max_instances_per_category) - 1
            masks.append(mask.astype(np.uint8))
            labels.append(label)
            ids.append(int(pann_value))

        height, width = panoptic.shape
        sample[self._masks_key] = _stack_masks(masks, height, width)
        sample[self._labels_key] = _target_to_array(labels)
        sample[self._ids_key] = _target_to_array(ids)

        return sample, {}
