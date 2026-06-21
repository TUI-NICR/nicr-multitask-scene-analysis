# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torchmetrics import ConfusionMatrix

from ...data.preprocessing.resize import get_fullres
from ...data.preprocessing.resize import get_fullres_key
from ...loss import CosineEmbeddingLoss
from ...metric import MeanIntersectionOverUnion
from ...metric.mae import PanopticQualityWithOrientationMAE
from ...types import BatchType
from ...visualization import PanopticColorGenerator
from ...visualization import visualize_panoptic_pil
from ...visualization import visualize_semantic_pil
from ..base import append_detached_losses_to_logs
from ..base import append_profile_to_logs
from .base import TokenMatchingCache
from .base import TokenMatchingTaskHelperBase
from .base import TokenTaskHelperBase


class TokenEmbeddingTaskHelper(TokenMatchingTaskHelperBase):
    def __init__(
        self,
        *,
        matching_cache: TokenMatchingCache,
        n_classes: int,
        semantic_classes_is_thing: Sequence[bool],
        max_instances_per_category: int = 1 << 16,
        pq_n_workers: Optional[int] = None,
        examples_cmap: Optional[Sequence[Tuple[int, int, int]]] = None,
        norm_margin: float = 1.0,
        norm_coefficient: float = 5e-2,
        embedding_coefficient: float = 2.0,
        negative_coefficient: float = 0.0,
        negative_margin: float = 0.0,
        disable_aux_loss: bool = False,
        linear_probing_coefficient: float = 1.0,
        enable_linear_probing: bool = False
    ):
        super().__init__(matching_cache=matching_cache)
        self._n_classes = n_classes
        self._classes_is_thing = tuple(
            bool(v) for v in semantic_classes_is_thing
        )
        self._max_instances_per_category = max_instances_per_category
        self._pq_n_workers = pq_n_workers
        self._examples: Dict[str, Any] = {}
        self._examples_cmap = examples_cmap
        self._norm_margin = norm_margin
        self._norm_coefficient = norm_coefficient
        self._embedding_coefficient = embedding_coefficient
        self._negative_coefficient = negative_coefficient
        self._negative_margin = negative_margin
        self._disable_aux_loss = disable_aux_loss
        self._linear_probing_coefficient = linear_probing_coefficient
        self._enable_linear_probing = enable_linear_probing

    def initialize(self, device: torch.device):
        self._loss = CosineEmbeddingLoss()
        # PQ metrics stay on CPU as their update runs in worker processes.
        if self._enable_linear_probing:
            self._linear_probing_loss = torch.nn.CrossEntropyLoss(
                ignore_index=-1,
                reduction='mean'
            )
            self._linear_probing_metric = MeanIntersectionOverUnion(
                n_classes=self._n_classes
            )
            self._linear_probing_metric.reset()
        self._text_metric = MeanIntersectionOverUnion(
            n_classes=self._n_classes
        )
        self._text_metric.reset()
        self._visual_metric = MeanIntersectionOverUnion(
            n_classes=self._n_classes
        )
        self._visual_metric.reset()
        assert len(self._classes_is_thing) == self._n_classes + 1
        if self._enable_linear_probing:
            self._linear_probing_pq_metric = PanopticQualityWithOrientationMAE(
                num_categories=self._n_classes + 1,
                ignored_label=0,
                max_instances_per_category=self._max_instances_per_category,
                offset=256**3,
                is_thing=self._classes_is_thing,
                num_workers=self._pq_n_workers
            )
        self._text_pq_metric = PanopticQualityWithOrientationMAE(
            num_categories=self._n_classes + 1,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=self._classes_is_thing,
            num_workers=self._pq_n_workers
        )
        self._visual_pq_metric = PanopticQualityWithOrientationMAE(
            num_categories=self._n_classes + 1,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=self._classes_is_thing,
            num_workers=self._pq_n_workers
        )
        if self._examples_cmap is not None:
            cmap = tuple(self._examples_cmap)
            self._examples_cmap = cmap
            self._color_generator = PanopticColorGenerator(
                classes_colors=cmap,
                classes_is_thing=self._classes_is_thing,
                max_instances=self._max_instances_per_category,
                void_label=0
            )
        self._cos_sum_per_class = torch.zeros(
            self._n_classes,
            dtype=torch.float32,
            device='cpu'
        )
        self._cos_count_per_class = torch.zeros(
            self._n_classes,
            dtype=torch.int64,
            device='cpu'
        )

    def _build_target_entries(
        self,
        *,
        batch: BatchType,
        embed_dim: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # collect the per-sample target embedding for every matched query.
        # the dataset preprocessor stores a LUT of reference embeddings per
        # image plus an index map. we resolve indices into embeddings here
        # and return one (n_segments, embed_dim) tensor per sample.
        if 'dense_visual_embedding_lut' not in batch:
            raise ValueError(
                "token-visual-embedding requires "
                "'dense_visual_embedding_lut' in the batch. Ensure "
                "preprocessing added DenseVisualEmbeddingTargetGenerator."
            )
        if 'dense_visual_embedding_indices' not in batch:
            raise ValueError(
                "token-visual-embedding requires "
                "'dense_visual_embedding_indices' in the batch. Ensure "
                "preprocessing added DenseVisualEmbeddingTargetGenerator."
            )
        if 'panoptic_token_masks' not in batch:
            raise ValueError(
                "token-visual-embedding requires 'panoptic_token_masks' in "
                "the batch."
            )

        luts = batch['dense_visual_embedding_lut']
        indices_per_sample = batch['dense_visual_embedding_indices']
        masks_per_sample = batch['panoptic_token_masks']
        assert len(luts) == len(indices_per_sample) == len(masks_per_sample)

        target_vectors = []
        target_valid = []
        for lut, indices, masks in zip(
            luts, indices_per_sample, masks_per_sample
        ):
            if masks.numel() == 0:
                vectors = torch.zeros((0, embed_dim), dtype=torch.float32)
                valid = torch.zeros((0,), dtype=torch.bool)
                target_vectors.append(vectors)
                target_valid.append(valid)
                continue
            if masks.dim() != 3:
                raise ValueError(
                    "panoptic_token_masks must be shaped [N, H, W]."
                )
            n_targets = masks.shape[0]
            vectors = torch.zeros((n_targets, embed_dim), dtype=torch.float32)
            valid = torch.zeros(n_targets, dtype=torch.bool)
            for idx in range(n_targets):
                mask = masks[idx].to(indices.device, dtype=torch.bool)
                lut_indices = indices[mask]
                lut_indices = lut_indices[lut_indices > 0]
                if lut_indices.numel() == 0:
                    continue
                vals, counts = torch.unique(lut_indices, return_counts=True)
                lut_idx = int(vals[counts.argmax()].item()) - 1
                if lut_idx < 0 or lut_idx >= lut.shape[0]:
                    continue
                vector_tensor = lut[lut_idx].to(dtype=torch.float32)
                assert vector_tensor.shape[-1] == embed_dim
                vectors[idx] = vector_tensor
                valid[idx] = True
            target_vectors.append(vectors)
            target_valid.append(valid)
        return target_vectors, target_valid

    def _compute_area_weighted_loss(
        self,
        *,
        embeddings: torch.Tensor,
        matching: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        target_vectors: Sequence[torch.Tensor],
        target_masks: Sequence[torch.Tensor],
        valid_entries: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, int]:
        # cosine-similarity loss between predicted query embeddings and the
        # matched target embeddings, weighted by target segment area so a
        # large surface contributes more than a small one.
        losses = []
        weights = []
        device = embeddings.device
        for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
            if src_indices.numel() == 0:
                continue
            valid = valid_entries[sample_idx].to(device)
            if valid.numel() == 0 or not valid.any():
                continue
            tgt_indices = tgt_indices.to(device)
            src_indices = src_indices.to(device)
            valid_tgt = valid[tgt_indices]
            if not valid_tgt.any():
                continue
            src = src_indices[valid_tgt]
            tgt = tgt_indices[valid_tgt]
            preds = embeddings[sample_idx, src]
            targets = target_vectors[sample_idx].to(device)[tgt]
            cos = torch.nn.functional.cosine_similarity(preds, targets, dim=-1)
            loss = 1.0 - cos
            masks = target_masks[sample_idx].to(device)[tgt]
            area = masks.flatten(1).sum(dim=1).float()
            weight = torch.sqrt(area)
            losses.append(loss)
            weights.append(weight)

        if not losses:
            return embeddings.sum() * 0.0, 0

        loss_vec = torch.cat(losses, dim=0)
        weight_vec = torch.cat(weights, dim=0)
        mean_weight = weight_vec.mean().clamp_min(1e-6)
        weight_vec = (weight_vec / mean_weight).clamp(0.5, 4.0)
        return (loss_vec * weight_vec).sum(), int(weight_vec.numel())

    def _compute_negative_loss(
        self,
        *,
        embeddings: torch.Tensor,
        matching: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        target_vectors: Sequence[torch.Tensor],
        target_masks: Sequence[torch.Tensor],
        valid_entries: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, int]:
        # pushes each matched query away from non-matching targets within
        # the sample, using a margin so well-separated pairs don't pay any
        # loss. only fires when `--token-visual-embedding-negative-coefficient`
        # is non-zero; otherwise we return a zero loss that still carries a
        # gradient to keep DDP happy.
        if self._negative_coefficient <= 0.0:
            return embeddings.sum() * 0.0, 0
        losses = []
        weights = []
        device = embeddings.device
        for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
            if src_indices.numel() < 2:
                continue
            valid = valid_entries[sample_idx].to(device)
            if valid.numel() == 0 or not valid.any():
                continue
            tgt_indices = tgt_indices.to(device)
            src_indices = src_indices.to(device)
            valid_tgt = valid[tgt_indices]
            if valid_tgt.sum().item() < 2:
                continue
            src = src_indices[valid_tgt]
            tgt = tgt_indices[valid_tgt]
            preds = embeddings[sample_idx, src]
            targets = target_vectors[sample_idx].to(device)[tgt]
            targets = torch.roll(targets, shifts=1, dims=0)
            cos = torch.nn.functional.cosine_similarity(preds, targets, dim=-1)
            loss = torch.relu(cos - self._negative_margin)
            masks = target_masks[sample_idx].to(device)[tgt]
            area = masks.flatten(1).sum(dim=1).float()
            weight = torch.sqrt(area)
            losses.append(loss)
            weights.append(weight)

        if not losses:
            return embeddings.sum() * 0.0, 0

        loss_vec = torch.cat(losses, dim=0)
        weight_vec = torch.cat(weights, dim=0)
        mean_weight = weight_vec.mean().clamp_min(1e-6)
        weight_vec = (weight_vec / mean_weight).clamp(0.5, 4.0)
        return (loss_vec * weight_vec).sum(), int(weight_vec.numel())


    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # main loss assembly. aggregates over all decoder stages: the
        # cosine area-weighted positive loss against matched targets,
        # the optional negative (push-away) loss, a per-class-mean
        # attractor, an embedding-norm regulariser, and the supervised
        # linear-probing CE head if present.
        embeddings = predictions_post['token_visual_embedding_output']
        side_outputs = predictions_post['token_visual_embedding_side_outputs']
        if self._disable_aux_loss:
            side_outputs = ()
        loss_dict = {}
        stage_losses = []
        norm_stage_losses = []
        aux_keys = tuple(f'aux_{idx+1}' for idx in range(len(side_outputs)))
        keys = aux_keys + ('main',)
        stage_embeddings = side_outputs + (embeddings,)
        vector_entries, valid_entries = self._build_target_entries(
            batch=batch,
            embed_dim=embeddings.shape[2]
        )
        target_labels = batch['panoptic_token_labels']
        target_masks = batch['panoptic_token_masks']
        linear_logits = predictions_post.get(
            'token_visual_embedding_linear_probing_logits'
        )
        if linear_logits is not None and not self._enable_linear_probing:
            linear_logits = None
        for key, tensor in zip(keys, stage_embeddings):
            matching = self.matching_cache.get_stage(batch_idx, key)
            stage_loss_raw, n_elements = self._compute_area_weighted_loss(
                embeddings=tensor,
                matching=matching,
                target_vectors=vector_entries,
                target_masks=target_masks,
                valid_entries=valid_entries
            )
            stage_loss = stage_loss_raw / max(n_elements, 1)
            stage_loss = stage_loss * self._embedding_coefficient
            loss_key = 'token_visual_embedding_loss'
            if key != 'main':
                loss_key = f'token_visual_embedding_loss_{key}'
            loss_dict[loss_key] = stage_loss
            stage_losses.append(stage_loss)
            neg_loss_raw, neg_elements = self._compute_negative_loss(
                embeddings=tensor,
                matching=matching,
                target_vectors=vector_entries,
                target_masks=target_masks,
                valid_entries=valid_entries
            )
            if neg_elements > 0:
                neg_loss = (
                    (neg_loss_raw / neg_elements)
                    * self._negative_coefficient
                )
            else:
                neg_loss = neg_loss_raw * 0.0
            loss_dict[f'token_visual_embedding_loss_negative_{key}'] = neg_loss
            stage_losses.append(neg_loss)
            # A void vector is not well-defined for unmatched queries, so we
            # sink their embedding magnitude instead.
            norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1)
            matched = torch.zeros_like(norms, dtype=torch.bool)
            for sample_idx, (src_indices, _) in enumerate(matching):
                if src_indices.numel() > 0:
                    matched[sample_idx, src_indices] = True
            # The L2 magnitude is pushed above the margin for matched queries
            # and toward zero for unmatched queries.
            pos = torch.relu(self._norm_margin - norms).pow(2)
            neg = norms.pow(2)
            if matched.any():
                pos_loss = pos[matched].mean()
            else:
                pos_loss = norms.sum() * 0.0
            if (~matched).any():
                neg_loss = neg[~matched].mean()
            else:
                neg_loss = norms.sum() * 0.0
            norm_loss = self._norm_coefficient * (pos_loss + neg_loss)
            loss_dict[f'token_visual_embedding_loss_norm_{key}'] = norm_loss
            norm_stage_losses.append(norm_loss)
        if stage_losses:
            total_loss = torch.sum(torch.stack(stage_losses))
        else:
            total_loss = embeddings.sum() * 0.0
        if norm_stage_losses:
            total_loss = total_loss + torch.sum(torch.stack(norm_stage_losses))
        if linear_logits is not None and self._enable_linear_probing:
            matching = self.matching_cache.get_stage(batch_idx, 'main')
            n_logits = linear_logits.shape[-1]
            matched_linear_logits = []
            target_classes = []
            for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
                if src_indices.numel() == 0:
                    continue
                labels = target_labels[sample_idx]
                if labels is None or labels.numel() == 0:
                    continue
                valid = (labels >= 0) & (labels < n_logits)
                if not valid.any():
                    continue
                tgt_indices = tgt_indices.to(labels.device)
                src_indices = src_indices.to(linear_logits.device)
                valid_tgt = valid[tgt_indices]
                if not valid_tgt.any():
                    continue
                matched_linear_logits.append(
                    linear_logits[sample_idx, src_indices[valid_tgt]]
                )
                target_classes.append(labels[tgt_indices[valid_tgt]])
            if matched_linear_logits:
                ce_loss = self._linear_probing_loss(
                    torch.cat(matched_linear_logits, dim=0),
                    torch.cat(target_classes, dim=0)
                ) * 10
            else:
                ce_loss = linear_logits.sum() * 0.0
            ce_loss = ce_loss * self._linear_probing_coefficient
            loss_dict['token_visual_embedding_linear_probing_loss'] = ce_loss
            total_loss = total_loss + ce_loss
        loss_dict[self.mark_as_total('token_visual_embedding')] = total_loss
        return loss_dict

    def _update_cosine_stats(
        self,
        *,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType,
        target_vectors: Sequence[torch.Tensor],
        valid_entries: Sequence[torch.Tensor]
    ) -> None:
        embeddings = predictions_post['token_visual_embedding_output']
        target_labels = batch['panoptic_token_labels']
        matching = self.matching_cache.get_stage(batch_idx, 'main')
        device = embeddings.device
        for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
            if src_indices.numel() == 0:
                continue
            labels = target_labels[sample_idx]
            if labels is None or labels.numel() == 0:
                continue
            valid = valid_entries[sample_idx].to(device)
            if valid.numel() == 0 or not valid.any():
                continue
            tgt_indices = tgt_indices.to(device)
            src_indices = src_indices.to(device)
            valid_tgt = valid[tgt_indices]
            if not valid_tgt.any():
                continue
            tgt = tgt_indices[valid_tgt]
            src = src_indices[valid_tgt]
            label_vals = labels[tgt].to(device)
            preds = embeddings[sample_idx, src]
            targets = target_vectors[sample_idx].to(device)[tgt]
            cos = torch.nn.functional.cosine_similarity(preds, targets, dim=-1)
            for cls in torch.unique(label_vals):
                if cls < 0 or cls >= self._n_classes:
                    continue
                cls_mask = label_vals == cls
                if not cls_mask.any():
                    continue
                cls_idx = int(cls.item())
                self._cos_sum_per_class[cls_idx] += (
                    cos[cls_mask].sum().detach().cpu()
                )
                self._cos_count_per_class[cls_idx] += (
                    cls_mask.sum().detach().cpu()
                )

    @append_profile_to_logs('token_visual_embedding_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        return loss_dict, {}

    @append_profile_to_logs('token_visual_embedding_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # in addition to the training losses, gather metrics: cosine
        # similarity stats against targets, per-source semantic mIoU
        # (text / visual-mean / linear-probing), and per-source panoptic
        # quality if class is-thing info is available.
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        embeddings = predictions_post['token_visual_embedding_output']
        vector_entries, valid_entries = self._build_target_entries(
            batch=batch,
            embed_dim=embeddings.shape[2]
        )
        self._update_cosine_stats(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post,
            target_vectors=vector_entries,
            valid_entries=valid_entries
        )
        target = get_fullres(batch, 'semantic')
        target_tensor = torch.as_tensor(target)
        mask = (target_tensor != 0).to(dtype=torch.bool)
        target_flat = (target_tensor[mask] - 1).reshape(-1)

        key_text_fullres = get_fullres_key(
            'token_visual_embedding_text_based_semantic_idx'
        )
        preds_text = predictions_post[key_text_fullres]
        if preds_text is not None:
            preds_tensor = torch.as_tensor(preds_text).to(mask.device)
            assert preds_tensor.shape == mask.shape
            preds_flat = preds_tensor[mask].reshape(-1)
            metric_device = self._text_metric.confmat.device
            self._text_metric.update(
                preds=preds_flat.to(metric_device),
                target=target_flat.to(metric_device)
            )

        key_visual_fullres = get_fullres_key(
            'token_visual_embedding_visual_mean_based_semantic_idx'
        )
        preds_visual = predictions_post[key_visual_fullres]
        if preds_visual is not None:
            preds_tensor = torch.as_tensor(preds_visual).to(mask.device)
            assert preds_tensor.shape == mask.shape
            preds_flat = preds_tensor[mask].reshape(-1)
            metric_device = self._visual_metric.confmat.device
            self._visual_metric.update(
                preds=preds_flat.to(metric_device),
                target=target_flat.to(metric_device)
            )
        if self._enable_linear_probing:
            key_linear_fullres = get_fullres_key(
                'token_visual_embedding_linear_probing_semantic_idx'
            )
            preds_linear = predictions_post.get(key_linear_fullres)
            if preds_linear is not None:
                preds_tensor = torch.as_tensor(preds_linear).to(mask.device)
                assert preds_tensor.shape == mask.shape
                preds_flat = preds_tensor[mask].reshape(-1)
                metric_device = self._linear_probing_metric.confmat.device
                self._linear_probing_metric.update(
                    preds=preds_flat.to(metric_device),
                    target=target_flat.to(metric_device)
                )
        if batch_idx == 0 and self._examples_cmap is not None:
            self._append_examples(batch_idx, predictions_post)
        self._update_panoptic_metrics(
            batch=batch,
            predictions_post=predictions_post,
            prefix='token_visual_embedding_text_based',
            metric=self._text_pq_metric
        )
        self._update_panoptic_metrics(
            batch=batch,
            predictions_post=predictions_post,
            prefix='token_visual_embedding_visual_mean_based',
            metric=self._visual_pq_metric
        )
        if self._enable_linear_probing:
            self._update_panoptic_metrics(
                batch=batch,
                predictions_post=predictions_post,
                prefix='token_visual_embedding_linear_probing',
                metric=self._linear_probing_pq_metric
            )

        return loss_dict, {}

    def _update_panoptic_metrics(
        self,
        *,
        batch: BatchType,
        predictions_post: BatchType,
        prefix: str,
        metric: Optional[PanopticQualityWithOrientationMAE]
    ) -> None:
        if metric is None:
            return
        panoptic_key = f'{prefix}_panoptic_segmentation'
        fullres_key = get_fullres_key(panoptic_key)
        panoptic_seg = predictions_post.get(fullres_key)
        if panoptic_seg is None:
            panoptic_seg = predictions_post.get(panoptic_key)
        panoptic_ids = predictions_post.get(f'{prefix}_panoptic_id_dicts')
        if panoptic_seg is None or panoptic_ids is None:
            return
        assert isinstance(panoptic_seg, torch.Tensor)
        target_panoptic = get_fullres(batch, 'panoptic')
        assert target_panoptic is not None, (
            "TokenEmbeddingTaskHelper expects 'panoptic_fullres' entries in "
            'the batch.'
        )
        panoptic_targets = target_panoptic.to(torch.int64).cpu()
        if panoptic_seg.shape[-2:] != panoptic_targets.shape[-2:]:
            panoptic_seg = torch.nn.functional.interpolate(
                panoptic_seg.float().unsqueeze(1),
                size=panoptic_targets.shape[-2:],
                mode='nearest',
            ).squeeze(1).to(torch.int64)
        panoptic_list = [tensor.detach().cpu() for tensor in panoptic_seg]
        stacked = torch.stack(panoptic_list).to(torch.int64)
        target_ids = batch['panoptic_ids_to_instance_dict']
        metric.update(
            panoptic_preds=stacked,
            orientation_preds=None,
            panoptic_preds_id_dicts=panoptic_ids,
            panoptic_target=panoptic_targets,
            orientation_target=None,
            panoptic_target_id_dicts=target_ids
        )

    def validation_epoch_end(self):
        artifacts: Dict[str, torch.Tensor] = {}
        examples: Dict[str, Any] = {}
        logs: Dict[str, torch.Tensor] = {}
        miou, ious = self._text_metric.compute(return_ious=True)
        logs['token_visual_embedding_text_based_miou'] = miou.cpu()
        artifacts['token_visual_embedding_text_based_cm'] = \
            self._text_metric.confmat.clone().cpu()
        artifacts['token_visual_embedding_text_based_ious_per_class'] = \
            ious.clone().cpu()
        self._text_metric.reset()

        miou, ious = self._visual_metric.compute(return_ious=True)
        logs['token_visual_embedding_visual_mean_based_miou'] = miou.cpu()
        artifacts['token_visual_embedding_visual_mean_based_cm'] = \
            self._visual_metric.confmat.clone().cpu()
        key = 'token_visual_embedding_visual_mean_based_ious_per_class'
        artifacts[key] = ious.clone().cpu()
        self._visual_metric.reset()

        pq_result = self._text_pq_metric.compute(suffix='_token')
        for key, value in pq_result.items():
            full_key = f'token_visual_embedding_text_based_{key}'
            if value.numel() == 1:
                logs[full_key] = value
            else:
                artifacts[full_key] = value
        self._text_pq_metric.reset()

        pq_result = self._visual_pq_metric.compute(suffix='_token')
        for key, value in pq_result.items():
            full_key = f'token_visual_embedding_visual_mean_based_{key}'
            if value.numel() == 1:
                logs[full_key] = value
            else:
                artifacts[full_key] = value
        self._visual_pq_metric.reset()

        if self._enable_linear_probing:
            miou, ious = self._linear_probing_metric.compute(return_ious=True)
            logs['token_visual_embedding_linear_probing_miou'] = miou.cpu()
            key = 'token_visual_embedding_linear_probing_ious_per_class'
            artifacts[key] = ious.clone().cpu()
            self._linear_probing_metric.reset()

        counts = self._cos_count_per_class.to(
            dtype=torch.float32
        ).clamp_min(1.0)
        mean_cos = self._cos_sum_per_class / counts
        artifacts['token_visual_embedding_cosine_distance_per_class'] = \
            (1.0 - mean_cos)
        artifacts['token_visual_embedding_cosine_count_per_class'] = \
            self._cos_count_per_class.clone()
        self._cos_sum_per_class.zero_()
        self._cos_count_per_class.zero_()

        if self._enable_linear_probing:
            pq_result = self._linear_probing_pq_metric.compute(suffix='_token')
            for key, value in pq_result.items():
                full_key = f'token_visual_embedding_linear_probing_{key}'
                if value.numel() == 1:
                    logs[full_key] = value
                else:
                    artifacts[full_key] = value
            self._linear_probing_pq_metric.reset()
        examples = dict(self._examples)
        self._examples = {}
        return artifacts, examples, logs

    def _append_examples(
        self,
        batch_idx: int,
        predictions_post: BatchType
    ) -> None:
        prefixes = (
            'token_visual_embedding_text_based',
            'token_visual_embedding_visual_mean_based',
            'token_visual_embedding_linear_probing'
        )
        for prefix in prefixes:
            sem_key = get_fullres_key(f'{prefix}_semantic_idx')
            if sem_key in predictions_post:
                sem_img = predictions_post[sem_key][0].cpu().numpy()
                key = f'{prefix}_semantic_example_batch_idx_{batch_idx}_0'
                colors = (
                    self._examples_cmap[1:]
                    if self._examples_cmap else None
                )
                self._examples[key] = visualize_semantic_pil(
                    sem_img,
                    colors=colors
                )
            pan_key = f'{prefix}_panoptic_segmentation'
            if pan_key in predictions_post:
                pan_img = predictions_post[pan_key][0].cpu().numpy()
                key = f'{prefix}_panoptic_example_batch_idx_{batch_idx}_0'
                self._examples[key] = visualize_panoptic_pil(
                    pan_img,
                    shared_color_generator=self._color_generator
                )


class TokenImageEmbeddingTaskHelper(TokenTaskHelperBase):
    def __init__(
        self,
        *,
        n_classes: int
    ):
        super().__init__()
        self._n_classes: int = n_classes

        # linear probing
        self._linear_probing_layers: Dict[str, torch.nn.Module] = {}

    def initialize(self, device: torch.device):
        self._loss = CosineEmbeddingLoss()
        self._text_metric_cm = ConfusionMatrix(
            task='multiclass',
            num_classes=self._n_classes,
            ignore_index=-1
        )
        self._text_metric_cm.reset()
        self._visual_metric_cm = ConfusionMatrix(
            task='multiclass',
            num_classes=self._n_classes,
            ignore_index=-1
        )
        self._visual_metric_cm.reset()

        self._ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            reduction='mean'
        )

        self._linear_probing_cm = ConfusionMatrix(
            task='multiclass',
            num_classes=self._n_classes,
            ignore_index=-1
        )
        self._linear_probing_cm.reset()

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        predicted_embeddings = predictions_post[
            'token_image_embedding_output'
        ]
        targets = batch['image_embedding']
        # Drop the singleton CLS-token axis: [B, 1, D] -> [B, D].
        if predicted_embeddings.ndim == 3:
            predicted_embeddings = predicted_embeddings.squeeze(1)

        loss_dict = {}
        (loss_raw, n_elements), = self._loss(
            input_tensors=(predicted_embeddings,),
            target_tensors=(targets,)
        )
        loss_dict['token_image_embedding_loss'] = (
            loss_raw / max(n_elements, 1)
        )
        raw_losses = [loss_raw]
        counts = [n_elements]

        scene_logits = predictions_post[
            'token_image_embedding_linear_probing_logits'
        ]
        if scene_logits is not None:
            ce_loss_per_dataset: Dict[str, torch.Tensor] = {}
            sample_count_per_dataset: Dict[str, int] = {}
            # we have a linear probing head so we add extra loss
            scene_class_targets = batch['scene'].long() - 1
            for sample_logits, sample_target, sample_meta in zip(
                scene_logits,
                scene_class_targets,
                batch['meta']
            ):
                dataset_name = sample_meta['dataset_type'].__name__

                ce_loss = self._ce_loss(
                    sample_logits,
                    sample_target.unsqueeze(0)
                )

                if torch.isnan(ce_loss):
                    continue

                if dataset_name not in ce_loss_per_dataset:
                    ce_loss_per_dataset[dataset_name] = ce_loss
                    sample_count_per_dataset[dataset_name] = 1
                else:
                    ce_loss_per_dataset[dataset_name] += ce_loss
                    sample_count_per_dataset[dataset_name] += 1

            for dataset_name, dataset_loss in ce_loss_per_dataset.items():
                count = sample_count_per_dataset[dataset_name]
                avg_loss = dataset_loss / max(count, 1)
                key = (
                    'token_image_embedding_linear_probing_loss_'
                    f'{dataset_name.lower()}'
                )
                loss_dict[key] = avg_loss
                # add to total loss
                raw_losses.append(dataset_loss)
                counts.append(1)

        total_loss = self.accumulate_losses(raw_losses, counts)

        loss_dict[self.mark_as_total('token_image_embedding')] = total_loss
        return loss_dict

    @append_profile_to_logs('token_image_embedding_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)

        return loss_dict, {}

    @append_profile_to_logs('token_image_embedding_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)

        assert all(
            batch['meta'][0]['dataset_type'] == sample_meta['dataset_type']
            for sample_meta in batch['meta']
        )

        # shift to remove void class
        targets = batch['scene'].long().cpu() - 1

        self._text_metric_cm.update(
            preds=predictions_post[
                'token_image_embedding_text_based_scene_idx'
            ].cpu(),
            target=targets
        )

        self._visual_metric_cm.update(
            preds=predictions_post[
                'token_image_embedding_visual_mean_based_scene_idx'
            ].cpu(),
            target=targets
        )

        linear_probing_idx = predictions_post.get(
            'token_image_embedding_linear_probing_idx',
            None
        )
        if linear_probing_idx is not None:
            self._linear_probing_cm.update(
                preds=linear_probing_idx.cpu(),
                target=targets.unsqueeze(0)
            )

        return loss_dict, {}

    def _compute_metrics_from_cm(
        self, cm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tp = torch.diag(cm)
        gt = torch.sum(cm, dim=1)
        # ignore empty classes
        tp = tp[gt != 0]
        gt = gt[gt != 0]
        acc = tp.sum().float() / gt.sum().float()
        bacc = torch.mean(tp.float() / gt.float())

        return acc, bacc

    def validation_epoch_end(self):
        artifacts: Dict[str, torch.Tensor] = {}
        examples: Dict[str, Any] = {}
        logs: Dict[str, torch.Tensor] = {}

        text_acc, text_bacc = self._compute_metrics_from_cm(
            self._text_metric_cm.confmat
        )

        visual_acc, visual_bacc = self._compute_metrics_from_cm(
            self._visual_metric_cm.confmat
        )

        artifacts['token_image_embedding_text_based_cm'] = (
            self._text_metric_cm.confmat.clone()
        )
        logs['token_image_embedding_text_based_acc'] = text_acc
        logs['token_image_embedding_text_based_bacc'] = text_bacc
        self._text_metric_cm.reset()

        artifacts['token_image_embedding_visual_mean_based_cm'] = (
            self._visual_metric_cm.confmat.clone()
        )
        logs['token_image_embedding_visual_mean_based_acc'] = visual_acc
        logs['token_image_embedding_visual_mean_based_bacc'] = visual_bacc
        self._visual_metric_cm.reset()

        artifacts['token_image_embedding_linear_probing_cm'] = (
            self._linear_probing_cm.confmat.clone()
        )
        linear_probing_acc, linear_probing_bacc = self._compute_metrics_from_cm(
            self._linear_probing_cm.confmat
        )
        logs['token_image_embedding_linear_probing_acc'] = linear_probing_acc
        logs['token_image_embedding_linear_probing_bacc'] = linear_probing_bacc
        self._linear_probing_cm.reset()

        return artifacts, examples, logs
