# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..metric.mae import PanopticQualityWithOrientationMAE
from ..metric import MeanIntersectionOverUnion
from ..types import BatchType
from .base import append_profile_to_logs
from .base import TaskHelperBase
from ..visualization import visualize_panoptic_pil
from ..visualization import visualize_instance_pil
from ..visualization import visualize_semantic_pil

from nicr_scene_analysis_datasets.dataset_base import SemanticLabelList


class PanopticTaskHelper(TaskHelperBase):
    def __init__(
        self,
        semantic_n_classes: int,     # with void!
        semantic_classes_is_thing: Tuple[bool],
        semantic_label_list: SemanticLabelList
    ) -> None:
        super().__init__()
        # required for panoptic quality
        self._semantic_n_classes = semantic_n_classes
        self._semantic_classes_is_thing = semantic_classes_is_thing
        self._semantic_label_list = semantic_label_list

        # Used for panopticapi visualization
        categories = []
        for idx, label in enumerate(self._semantic_label_list):
            label_dict = {}
            label_dict['supercategory'] = label.class_name
            label_dict['name'] = label.class_name
            label_dict['id'] = idx
            label_dict['isthing'] = int(label.is_thing)
            label_dict['color'] = [int(a) for a in label.color]
            categories.append(label_dict)
        categories = {cat['id']: cat for cat in categories}
        self._coco_categories = categories

        # during validation, we store some examples for visualization purposes
        self._examples = {}

        # hypersim has more than 256 instances per image
        self._max_instances_per_category = 1 << 16     # 1 << 8
        self._thing_ids = np.where(self._semantic_classes_is_thing)[0]
        self._with_orientation = False

    def initialize(self, device: torch.device):
        # keep metrics on cpu
        self._mae_pq_deeplab = PanopticQualityWithOrientationMAE(
            num_categories=self._semantic_n_classes,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=self._semantic_classes_is_thing
        )

        # metrics (keep it on cpu, it is faster)
        self._metric_iou = MeanIntersectionOverUnion(
            n_classes=self._semantic_n_classes,     # with void!
            ignore_first_class=True     # we ignore void pixels for miou
        )
        # self._metric_iou = self._metric_iou.to(device)
        self._metric_iou.reset()

    @append_profile_to_logs('panoptic_step_time')
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # For doing panoptic segmentation, the is no additional training_step
        # required.
        # The task helper only combines the semantic and instance segmentation
        # and calculates the PQ.
        return {}, {}

    @append_profile_to_logs('panoptic_step_time')
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self._with_orientation = 'orientations_present' in batch

        # get the orientation results
        if self._with_orientation:
            orientations_results = predictions_post['orientations_panoptic_segmentation_deeplab_instance_segmentation']
            orientations_targets = batch['orientations_present']
        else:
            orientations_results = None
            orientations_targets = None

        # panoptic_targets the target for panoptic segmentation
        panoptic_targets = get_fullres(batch, 'panoptic').cpu()
        panoptic_targets_id_dicts = batch['panoptic_ids_to_instance_dict']

        # panoptic segmentation as done in panoptic deeplab
        panoptic_deeplab_preds = predictions_post[get_fullres_key('panoptic_segmentation_deeplab')]
        panoptic_deeplab_id_dicts = predictions_post['panoptic_segmentation_deeplab_ids']
        panoptic_deeplab_preds = panoptic_deeplab_preds.cpu()
        self._mae_pq_deeplab.update(
            panoptic_preds=panoptic_deeplab_preds,
            orientation_preds=orientations_results,
            panoptic_preds_id_dicts=panoptic_deeplab_id_dicts,
            panoptic_target=panoptic_targets,
            orientation_target=orientations_targets,
            panoptic_target_id_dicts=panoptic_targets_id_dicts
        )

        # deeplab merging can change the semantic segmentation.
        # By diving the panoptic deeplab prediction by the max instances per category,
        # we can get this semantic segmentation back.
        deeplab_semantic = (panoptic_deeplab_preds // self._max_instances_per_category)
        target_semantic = get_fullres(batch, 'semantic')
        self._metric_iou.update(preds=deeplab_semantic.cpu(),
                                target=target_semantic.cpu())

        # store example for visualization
        if 0 == batch_idx:
            # panoptic segmentation
            deeplab_ex = panoptic_deeplab_preds[0].cpu()
            key = f'panoptic_example_batch_deeplab_{batch_idx}_0'
            self._examples[key] = visualize_panoptic_pil(
                deeplab_ex.numpy(),
                self._semantic_n_classes,
                self._coco_categories,
                self._max_instances_per_category
            )
            # semantic segmentation
            key = f'panoptic_example_batch_deeplab_semantic_{batch_idx}_0'
            self._examples[key] = visualize_semantic_pil(
                semantic_img=deeplab_semantic[0].cpu().numpy(),
                colors=self._semantic_label_list.colors_array
            )
            # instance segmentation
            instance_img = torch.zeros_like(panoptic_deeplab_preds[0])
            for idx, id_ in enumerate(torch.unique(panoptic_deeplab_preds[0])):
                mask = panoptic_deeplab_preds[0] == id_
                instance_img[mask] = idx

            instance_foreground = \
                predictions_post[get_fullres_key('panoptic_foreground_mask')].cpu()
            instance_img[~instance_foreground[0]] = 0
            key = f'panoptic_example_batch_deeplab_instance_{batch_idx}_0'
            self._examples[key] = visualize_instance_pil(instance_img.cpu().numpy())

        return {}, {}

    @append_profile_to_logs('panoptic_epoch_end_time')
    def validation_epoch_end(self):
        artifacts = {}
        logs = {}

        # panoptic segmentation as done in panoptic deeplab
        pq_deeplab_result = self._mae_pq_deeplab.compute(suffix="_deeplab")
        self._mae_pq_deeplab.reset()

        for key, value in pq_deeplab_result.items():
            if value.numel() == 1:
                logs[f'panoptic_{key}'] = value
            else:
                artifacts[f'panoptic_{key}'] = value

        # miou after panoptic merging as done in panoptic deeplab
        artifacts['panoptic_deeplab_semantic_cm'] = \
            self._metric_iou.confmat.clone()
        logs['panoptic_deeplab_semantic_miou'] = \
            self._metric_iou.compute()

        # reset metric (it is not done automatically)
        self._metric_iou.reset()
        assert self._metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs
