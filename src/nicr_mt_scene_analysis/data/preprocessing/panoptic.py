# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple, Union

import numpy as np

from ...data import CollateIgnoredDict
from ...types import BatchType
from ...utils.panoptic_merge import naive_merge_semantic_and_instance_np
from .base import PreprocessingBase
from .utils import _keys_available


class PanopticTargetGenerator(PreprocessingBase):
    def __init__(
        self,
        semantic_classes_is_thing: Union[None, Tuple[bool]] = None,  # with void
        use_is_thing_from_meta: bool = False,  # requires *-dataset package version >= 0.8.1
        multiscale_processing: bool = True,
    ) -> None:
        # convert list of booleans to list of thing class indices
        self._thing_class_ids = None
        # deprecated as it should be preferred to use the meta info
        if semantic_classes_is_thing is not None:
            assert not use_is_thing_from_meta
            self._thing_class_ids = np.where(semantic_classes_is_thing)[0]

        # store if is_thing list meta of batch should be used
        self._use_is_thing_from_meta = use_is_thing_from_meta

        if self._use_is_thing_from_meta:
            assert self._thing_class_ids is None, \
                "If use_is_thing_from_meta is True, semantic_classes_is_thing must not be set."

        # hypersim has more than 256 instances per image
        self._max_instances_per_category = (1 << 16)
        self._void_label = 0

        super().__init__(
            fixed_parameters={
                'max_instances_per_category': self._max_instances_per_category,
                'void_label': self._void_label
            },
            multiscale_processing=multiscale_processing
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        if not _keys_available(sample, ('instance', 'semantic')):
            # might be an inference call
            return sample, {}

        thing_ids = None
        if self._thing_class_ids is not None:
            thing_ids = self._thing_class_ids
        elif self._use_is_thing_from_meta:
            thing_ids = np.where(
                sample['meta']['semantic_label_list'].classes_is_thing
            )[0]

        semantic = sample['semantic']
        instance = sample['instance']

        panoptic_targets, panoptic_targets_id_dicts = \
            naive_merge_semantic_and_instance_np(
                semantic,
                instance,
                max_instances_per_category=self._max_instances_per_category,
                thing_ids=thing_ids,
                void_label=self._void_label
            )

        sample['panoptic'] = panoptic_targets
        # used for instance matching
        sample['panoptic_ids_to_instance_dict'] = \
            CollateIgnoredDict(panoptic_targets_id_dicts)

        return sample, {
            'thing_semantic_classes': thing_ids
        }
