# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import torch
import numpy as np

from ...data.preprocessing.resize import get_fullres_key
from ...data.preprocessing.resize import get_fullres_shape
from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .dense_base import DensePostProcessingBase
from .instance import InstancePostprocessing
from ...utils.panoptic_merge import deeplab_merge_batch
from .semantic import SemanticPostprocessing


class PanopticPostprocessing(DensePostProcessingBase):
    def __init__(
        self,
        semantic_postprocessing: SemanticPostprocessing,
        instance_postprocessing: InstancePostprocessing,
        semantic_classes_is_thing: Tuple[bool],
        semantic_class_has_orientation: Tuple[bool],
        normalized_offset: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        # store preprocessing objects
        self._semantic_postprocessing = semantic_postprocessing
        self._instance_postprocessing = instance_postprocessing

        # convert list of booleans to list of thing class indices
        self._thing_class_ids = np.where(semantic_classes_is_thing)[0]

        # +1 cause void class is missing in semantic_classes_is_thing
        self._thing_ids_panoptic = np.where(semantic_classes_is_thing)[0]+1
        self._orientation_ids = np.where(semantic_class_has_orientation)[0]+1

        self._normalized_offset = normalized_offset

        self._max_instances_per_category = 1 << 16

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs of panoptic helper
        (s_output, i_output), (s_side_outputs, i_side_outputs) = data

        # just call both postprocessing methods
        r_dict_semantic = self._semantic_postprocessing._postprocess_training(
            data=(s_output, s_side_outputs),
            batch=batch
        )
        r_dict_instance = self._instance_postprocessing._postprocess_training(
            data=(i_output, i_side_outputs),
            batch=batch
        )

        return {**r_dict_semantic, **r_dict_instance}

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs of panoptic helper
        (s_output, i_output), (s_side_outputs, i_side_outputs) = data

        # just call both postprocessing methods
        r_dict_semantic = self._semantic_postprocessing._postprocess_inference(
            data=(s_output, s_side_outputs),
            batch=batch
        )
        r_dict_instance = self._instance_postprocessing._postprocess_inference(
            data=(i_output, i_side_outputs),
            batch=batch
        )
        r_dict = {**r_dict_semantic, **r_dict_instance}

        # add additional postprocessing for panoptic segmentation
        # unpack instance decoder outputs
        with_orientation = (3 == len(i_output))
        if with_orientation:
            center_heatmap, center_offset, orientation = i_output
        else:
            center_heatmap, center_offset = i_output

        # undo normalizing if applied
        if self._normalized_offset:
            h, w = center_offset.shape[-2:]
            center_offset_ = center_offset.detach().clone()
            center_offset_[:, 0] = center_offset_[:, 0] * h
            center_offset_[:, 1] = center_offset_[:, 1] * w
        else:
            center_offset_ = center_offset

        # note that we compute the foreground mask before resizing to full
        # resolution
        # moreover, note that this mask does may not be valid for masking the
        # resulting panoptic segmentation, as panoptic merging may change
        # classes or instances
        semantic_segmentation = r_dict['semantic_segmentation_idx']
        foreground_mask = torch.isin(
            semantic_segmentation,
            test_elements=torch.tensor(self._thing_class_ids,
                                       device=semantic_segmentation.device)
        )
        r_dict['panoptic_foreground_mask'] = foreground_mask

        # get instance segmentation with panoptic foreground mask
        post = self._instance_postprocessing
        instance_predicted_centers, instance_segmentation = \
            post._get_instance_segmentation(center_heatmap,
                                            center_offset_,
                                            foreground_mask)
        r_dict['panoptic_instance_segmentation'] = \
            instance_segmentation
        r_dict['panoptic_instance_predicted_centers'] = \
            instance_predicted_centers

        # create panoptic segmentation using deeplab merging (put more focus on
        # instances and assign the a unique semantic class based on majority
        # vote)
        panoptic_seg, panoptic_ids = deeplab_merge_batch(
            (r_dict['semantic_segmentation_idx'].cpu() + 1),
            r_dict['panoptic_instance_segmentation'].cpu(),
            r_dict['panoptic_foreground_mask'].cpu(),
            self._max_instances_per_category,
            self._thing_ids_panoptic,
            0
        )
        r_dict['panoptic_segmentation_deeplab'] = panoptic_seg

        # derive semantic and instance segmention from panoptic segmentation
        # (note that deeplab merging may change the semantic classes)
        pan_semantic_seg = panoptic_seg // self._max_instances_per_category
        pan_semantic_ins = panoptic_seg % self._max_instances_per_category
        r_dict['panoptic_segmentation_deeplab_semantic_segmentation'] = pan_semantic_seg
        r_dict['panoptic_segmentation_deeplab_instance_segmentation'] = pan_semantic_ins

        # resize output to original shape (assume same shape for all samples)
        shape = get_fullres_shape(batch, 'instance')

        r_dict[get_fullres_key('panoptic_foreground_mask')] = \
            self._resize_prediction(
                foreground_mask,
                shape=shape,
                mode='nearest'
            )
        r_dict[get_fullres_key('panoptic_instance_segmentation')] = \
            self._resize_prediction(
                instance_segmentation,
                shape=shape,
                mode='nearest'
            )

        r_dict[get_fullres_key('panoptic_segmentation_deeplab')] = \
            self._resize_prediction(
                panoptic_seg,
                shape=shape,
                mode='nearest'
            )

        # store matched panoptic ids for orientation matching later
        r_dict['panoptic_segmentation_deeplab_ids'] = panoptic_ids

        # orientation estimation
        if with_orientation:
            # orientation fg mask
            foreground_mask_orientation = torch.isin(
                pan_semantic_seg,
                test_elements=torch.tensor(self._orientation_ids,
                                           device=pan_semantic_seg.device)
            )
            instance_segmentation_orientation = \
                r_dict['panoptic_instance_segmentation']

            r_dict['orientations_panoptic_segmentation_deeplab_instance_segmentation'] = \
                post._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=instance_segmentation_orientation,
                    foreground_mask=foreground_mask_orientation
                )

        return r_dict
