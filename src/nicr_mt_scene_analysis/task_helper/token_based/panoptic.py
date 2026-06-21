# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from ...data.preprocessing.resize import get_fullres
from ...loss import CrossEntropyLossSemantic
from ...metric import MeanIntersectionOverUnion
from ...metric.mae import PanopticQualityWithOrientationMAE
from ...types import BatchType
from ...visualization import PanopticColorGenerator
from ...visualization import visualize_panoptic_pil
from ...visualization import visualize_semantic_pil
from ...data.preprocessing.resize import get_fullres_key
from ...model.postprocessing import TokenSemanticPostprocessing
from ..base import append_detached_losses_to_logs
from ..base import append_profile_to_logs
from .base import TokenMatchingCache
from .base import TokenMatchingTaskHelperBase


class TokenPanopticTaskHelper(TokenMatchingTaskHelperBase):
    def __init__(
        self,
        *,
        semantic_n_classes_without_void: int,
        semantic_classes_is_thing: Tuple[bool, ...],
        matching_cache: TokenMatchingCache,
        mask_threshold: float = 0.8,
        mask_pixel_threshold: float = 0.1,
        overlap_threshold: float = 0.8,
        max_instances_per_category: int = 1 << 16,
        pq_n_workers: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        examples_cmap: Optional[Sequence[Tuple[int, int, int]]] = None,
        class_coefficient: float = 2.0,
        norm_margin: float = 1.0,
        norm_coefficient: float = 5e-2
    ):
        super().__init__(matching_cache=matching_cache)
        self._semantic_n_classes_without_void = semantic_n_classes_without_void
        self._classes_is_thing = semantic_classes_is_thing
        self._mask_threshold = mask_threshold
        self._mask_pixel_threshold = mask_pixel_threshold
        self._overlap_threshold = overlap_threshold
        self._max_instances_per_category = max_instances_per_category
        self._pq_n_workers = pq_n_workers
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._class_coefficient = class_coefficient
        self._norm_margin = norm_margin
        self._norm_coefficient = norm_coefficient
        # Optional dataset colors used only for validation examples.
        self._examples: Dict[str, Any] = {}
        self._examples_cmap = examples_cmap

    def initialize(self, device: torch.device):
        weights = None
        if self._class_weights is not None:
            weights = torch.as_tensor(
                self._class_weights,
                dtype=torch.float,
                device=device
            )
            if weights.shape[0] == self._semantic_n_classes_without_void + 1:
                weights = weights[1:]
            elif weights.shape[0] != self._semantic_n_classes_without_void:
                raise ValueError(
                    "Expected panoptic class weights with "
                    f"{self._semantic_n_classes_without_void} entries "
                    "(without void) but received "
                    f"{weights.shape[0]}."
                )
        self._weighted_reduction = weights is not None
        self._class_loss = CrossEntropyLossSemantic(
            weights=weights,
            label_smoothing=self._label_smoothing,
            weighted_reduction=self._weighted_reduction
        )
        # PQ has to stay on CPU: its update runs in worker processes and
        # returns CPU tensors. Keep panoptic mIoU on CPU as well because it is
        # derived from the same detached CPU panoptic predictions.
        is_thing = tuple(bool(v) for v in self._classes_is_thing)
        assert len(is_thing) == self._semantic_n_classes_without_void + 1
        self._pq_metric = PanopticQualityWithOrientationMAE(
            num_categories=self._semantic_n_classes_without_void + 1,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=is_thing,
            num_workers=self._pq_n_workers
        )
        self._metric_iou = MeanIntersectionOverUnion(
            n_classes=self._semantic_n_classes_without_void + 1,
            ignore_first_class=True
        )
        self._metric_iou.reset()
        if self._examples_cmap is not None:
            cmap = tuple(self._examples_cmap)
            self._examples_cmap = cmap
            self._color_generator = PanopticColorGenerator(
                classes_colors=cmap,
                classes_is_thing=is_thing,
                max_instances=self._max_instances_per_category,
                void_label=0
            )
        self._semantic_post = TokenSemanticPostprocessing(
            n_classes=self._semantic_n_classes_without_void
        )
        self._semantic_metric = MeanIntersectionOverUnion(
            n_classes=self._semantic_n_classes_without_void
        )
        self._semantic_metric.reset()

    def _compute_class_loss(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # weighted cross-entropy on the per-query class logits, using the
        # query to target assignment populated by TokenMaskTaskHelper. all
        # auxiliary decoder stages contribute under aux_* keys and the final
        # logits contribute under the `main` key.
        logits = predictions_post['token_semantic_output']
        side_outputs = predictions_post['token_semantic_side_outputs']
        stage_logits = side_outputs + (logits,)
        aux_keys = tuple(f'aux_{idx+1}' for idx in range(len(side_outputs)))
        keys = aux_keys + ('main',)
        target_entries = tuple(batch['panoptic_token_labels'])
        stage_losses = []
        norm_stage_losses = []
        loss_dict = {}
        for key, stage_logits in zip(keys, stage_logits):
            matching = self.matching_cache.get_stage(batch_idx, key)
            target_classes = []
            matched_class_logits = []
            for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
                # no targets in this sample, e.g. an all-void crop
                if src_indices.numel() == 0:
                    continue
                labels = target_entries[sample_idx]
                tgt_indices = tgt_indices.to(labels.device)
                src_indices = src_indices.to(stage_logits.device)
                matched_logits = stage_logits[
                    sample_idx,
                    src_indices
                ]
                matched_class_logits.append(matched_logits)
                target_classes.append(labels[tgt_indices])
            if matched_class_logits:
                # Panoptic token labels are zero-based. Add the void offset
                # here so CrossEntropyLossSemantic can shift them back to
                # torch CE targets internally.
                stage_loss, n_elements = self._class_loss(
                    input_tensors=(torch.cat(matched_class_logits, dim=0),),
                    target_tensors=(torch.cat(target_classes, dim=0) + 1,)
                )[0]
                if not self._weighted_reduction:
                    stage_loss = stage_loss / n_elements
            else:
                stage_loss = stage_logits.sum() * 0.0
            # Weight the supervised per-query class loss.
            stage_loss = stage_loss * self._class_coefficient
            loss_dict[f'token_panoptic_loss_class_{key}'] = stage_loss
            stage_losses.append(stage_loss)
            # DETR-style query decoders often handle unmatched queries with a
            # learned void class in CE. For embedding-style targets, a void
            # vector is not well-defined, so we instead sink the magnitude of
            # unmatched query predictions to treat it like token-visual-
            # embedding.
            norms = torch.linalg.vector_norm(stage_logits, ord=2, dim=-1)
            matched = torch.zeros_like(norms, dtype=torch.bool)
            for sample_idx, (src_indices, _) in enumerate(matching):
                if src_indices.numel() > 0:
                    matched[sample_idx, src_indices] = True
            # The L2 magnitude is pushed above the margin for matched queries
            # and toward zero for unmatched queries.
            pos = F.relu(self._norm_margin - norms).pow(2)
            neg = norms.pow(2)
            if matched.any():
                pos_loss = pos[matched].mean()
            else:
                pos_loss = norms.sum() * 0.0
            if (~matched).any():
                neg_loss = neg[~matched].mean()
            else:
                neg_loss = norms.sum() * 0.0
            # Weight the auxiliary logit-norm regularization.
            norm_loss = self._norm_coefficient * (pos_loss + neg_loss)
            loss_dict[f'token_panoptic_loss_norm_{key}'] = norm_loss
            norm_stage_losses.append(norm_loss)
        if stage_losses:
            class_loss_total = torch.sum(torch.stack(stage_losses))
        else:
            class_loss_total = logits.sum() * 0.0
        if norm_stage_losses:
            norm_loss_total = torch.sum(torch.stack(norm_stage_losses))
        else:
            norm_loss_total = logits.sum() * 0.0
        loss_dict['token_panoptic_loss_class'] = class_loss_total
        loss_dict['token_panoptic_loss_norm'] = norm_loss_total
        loss_dict[self.mark_as_total('token_panoptic')] = (
            class_loss_total + norm_loss_total
        )
        return loss_dict

    @append_profile_to_logs('token_panoptic_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_class_loss(batch, batch_idx, predictions_post)
        return loss_dict, {}

    @append_profile_to_logs('token_panoptic_step_time')
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # update the panoptic-quality metric from the assembled panoptic
        # prediction. additionally compute the semantic mIoU and the
        # validation cross-entropy loss when the semantic head is available.
        panoptic_key = 'token_panoptic_segmentation'
        fullres_key = get_fullres_key(panoptic_key)
        panoptic_seg = predictions_post.get(fullres_key)
        if panoptic_seg is None:
            panoptic_seg = predictions_post[panoptic_key]
        panoptic_ids = predictions_post['token_panoptic_id_dicts']
        assert isinstance(panoptic_seg, torch.Tensor)
        panoptic_list = [tensor.detach().cpu() for tensor in panoptic_seg]
        self._update_metrics(batch, panoptic_list, panoptic_ids)
        loss_dict = self._compute_class_loss(batch, batch_idx, predictions_post)
        if (
            self._semantic_post is not None
            and self._semantic_metric is not None
            and 'token_semantic_output' in predictions_post
        ):
            if 'token_mask_probs' not in predictions_post:
                if 'token_mask_output' not in predictions_post:
                    return loss_dict, {}
                mask_probs = predictions_post['token_mask_output'].sigmoid()
                predictions_post = dict(predictions_post)
                predictions_post['token_mask_probs'] = mask_probs
            semantic_dict = self._semantic_post._postprocess_inference(
                (predictions_post['token_semantic_output'],
                 predictions_post['token_semantic_side_outputs']),
                batch,
                out_dict={
                    'token_mask_probs': predictions_post['token_mask_probs'],
                }
            )
            target_semantic = get_fullres(batch, 'semantic')
            if target_semantic is not None:
                target_semantic = target_semantic.cpu()
                mask = target_semantic != 0
                preds_fullres = semantic_dict[
                    get_fullres_key('token_semantic_dense_idx')
                ]
                preds = preds_fullres[mask]
                target = target_semantic[mask] - 1
                self._semantic_metric.update(
                    preds=preds.cpu(),
                    target=target.cpu()
                )
        if batch_idx == 0 and self._examples_cmap is not None:
            pred_panoptic = panoptic_list[0]
            key = f'token_panoptic_example_batch_idx_{batch_idx}_0'
            self._examples[key] = visualize_panoptic_pil(
                pred_panoptic.cpu().numpy(),
                shared_color_generator=self._color_generator
            )
            pred_semantic = pred_panoptic // self._max_instances_per_category
            key = f'token_panoptic_example_batch_semantic_{batch_idx}_0'
            self._examples[key] = visualize_semantic_pil(
                semantic_img=pred_semantic.cpu().numpy(),
                colors=self._examples_cmap
            )
        return loss_dict, {}

    def _update_metrics(
        self,
        batch: BatchType,
        predictions: List[torch.Tensor],
        pred_ids: List[Dict[int, int]]
    ) -> None:
        stacked = torch.stack(predictions).to(torch.int64)
        # update PQ metric
        target_panoptic = get_fullres(batch, 'panoptic')
        assert target_panoptic is not None, (
            "TokenPanopticTaskHelper expects 'panoptic_fullres' entries in "
            'the batch.'
        )
        panoptic_targets = target_panoptic.to(torch.int64).cpu()
        target_ids = batch['panoptic_ids_to_instance_dict']
        self._pq_metric.update(
            panoptic_preds=stacked,
            orientation_preds=None,
            panoptic_preds_id_dicts=pred_ids,
            panoptic_target=panoptic_targets,
            orientation_target=None,
            panoptic_target_id_dicts=target_ids
        )

        # update mIoU metric
        pred_semantic = stacked // self._max_instances_per_category
        target_semantic = get_fullres(batch, 'semantic')
        assert target_semantic is not None, (
            "TokenPanopticTaskHelper expects 'semantic_fullres' entries in "
            'the batch.'
        )
        target_semantic = target_semantic.cpu()
        self._metric_iou.update(preds=pred_semantic, target=target_semantic)

    @append_profile_to_logs('token_panoptic_epoch_end_time')
    def validation_epoch_end(self):
        artifacts = {}
        logs = {}
        # compute PQ metric
        pq_result = self._pq_metric.compute(suffix='_token')
        for key, value in pq_result.items():
            full_key = f'token_panoptic_{key}'
            if value.numel() == 1:
                logs[full_key] = value
            else:
                artifacts[full_key] = value
        self._pq_metric.reset()

        # compute mIoU metric
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs['token_panoptic_miou'] = miou
        artifacts['token_panoptic_ious_per_class'] = ious.clone()
        self._metric_iou.reset()
        examples = dict(self._examples)
        if self._semantic_metric is not None:
            sem_miou, sem_ious = self._semantic_metric.compute(return_ious=True)
            logs['token_panoptic_semantic_miou'] = sem_miou
            key = 'token_panoptic_semantic_ious_per_class'
            artifacts[key] = sem_ious.clone()
            self._semantic_metric.reset()
        self._examples = {}
        return artifacts, examples, logs
