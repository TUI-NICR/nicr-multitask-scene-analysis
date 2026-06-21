# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nicr_scene_analysis_datasets.dataset_base import OrientationDict

from ...loss import VonMisesLossBiternion
from ...metric.mae import MeanAbsoluteAngularError
from ...types import BatchType
from ...utils import biternion2rad
from ...utils import np_rad2biternion
from ..base import append_detached_losses_to_logs
from ..base import append_profile_to_logs
from .base import TokenMatchingCache
from .base import TokenMatchingTaskHelperBase


class TokenOrientationTaskHelper(TokenMatchingTaskHelperBase):
    def __init__(
        self,
        *,
        matching_cache: TokenMatchingCache,
        supervise_confidence: bool = False,
        orientation_loss_weight: float = 1.0,
        confidence_loss_weight: float = 1.0,
        max_instances_per_category: int = 1 << 16,
        vonmises_kappa: float = 1.0
    ):
        super().__init__(matching_cache=matching_cache)
        self._supervise_confidence = supervise_confidence
        self._orientation_loss_weight = orientation_loss_weight
        self._confidence_loss_weight = confidence_loss_weight
        self._max_instances_per_category = max_instances_per_category
        self._vonmises_kappa = vonmises_kappa

    def _build_segment_orientation_targets(
        self,
        batch: BatchType,
        *,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[list, list]:
        panoptic_ids_per_sample = batch['panoptic_token_ids']
        orientations_per_sample = batch['orientations']
        vectors_list = []
        valid_list = []
        for panoptic_ids, orientations in zip(panoptic_ids_per_sample,
                                              orientations_per_sample):
            ids = [int(i) for i in panoptic_ids.tolist()]
            n_segments = len(ids)
            vectors = torch.zeros((n_segments, 2), dtype=dtype, device=device)
            valid = torch.zeros((n_segments,), dtype=torch.bool, device=device)
            for seg_idx, panoptic_id in enumerate(ids):
                instance_id = panoptic_id % self._max_instances_per_category
                if instance_id == 0 or instance_id not in orientations:
                    continue
                biternion = np_rad2biternion(
                    np.asarray(orientations[instance_id])
                )
                vectors[seg_idx] = torch.as_tensor(
                    biternion,
                    dtype=dtype,
                    device=device
                )
                valid[seg_idx] = True
            vectors_list.append(vectors)
            valid_list.append(valid)
        return vectors_list, valid_list

    def _build_query_orientation_targets(
        self,
        *,
        orientation_preds: torch.Tensor,
        segment_vectors: Sequence[torch.Tensor],
        segment_has_orientation: Sequence[torch.Tensor],
        matching: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_queries, _ = orientation_preds.shape
        assert len(segment_vectors) == len(segment_has_orientation)
        assert len(segment_vectors) == len(matching) == batch_size

        # Unmatched queries keep zero vectors and False orientation flags.
        query_vectors = torch.zeros_like(orientation_preds)
        query_has_orientation = torch.zeros(
            (batch_size, n_queries),
            dtype=torch.bool,
            device=orientation_preds.device
        )
        for sample_idx, (query_indices, segment_indices) in enumerate(matching):
            if query_indices.numel() == 0:
                continue
            # Matching maps query indices to panoptic segment indices.
            query_vectors[sample_idx, query_indices] = (
                segment_vectors[sample_idx][segment_indices]
            )
            query_has_orientation[sample_idx, query_indices] = (
                segment_has_orientation[sample_idx][segment_indices]
            )
        return query_vectors, query_has_orientation

    def initialize(self, device: torch.device):
        self._orientation_loss = VonMisesLossBiternion(
            kappa=self._vonmises_kappa,
            reduction='mean'
        )
        self._mae_metric = MeanAbsoluteAngularError()

    def _update_metrics(
        self,
        orientation_preds: torch.Tensor,
        targets: torch.Tensor,
        query_has_orientation: torch.Tensor
    ) -> None:
        preds_list = []
        targets_list = []
        batch_size = orientation_preds.shape[0]
        assert orientation_preds.shape[-1] == 2
        pred_angles = biternion2rad(
            orientation_preds.reshape(-1, 2)
        ).view(batch_size, orientation_preds.shape[1])
        target_angles = biternion2rad(
            targets.reshape(-1, 2)
        ).view(batch_size, targets.shape[1])
        for batch_idx in range(batch_size):
            pred_dict = OrientationDict()
            target_dict = OrientationDict()
            has_orientation = query_has_orientation[batch_idx].bool()
            token_ids = torch.nonzero(has_orientation, as_tuple=False).flatten()
            for token_id in token_ids:
                key = int(token_id.item())
                pred_dict[key] = float(pred_angles[batch_idx, key].item())
                target_dict[key] = float(target_angles[batch_idx, key].item())
            preds_list.append(pred_dict)
            targets_list.append(target_dict)
        self._mae_metric.update(preds_list, targets_list)

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        orientation_preds = predictions_post['token_orientation_queries']
        matching = self.matching_cache[batch_idx]
        segment_targets = self._build_segment_orientation_targets(
            batch,
            device=orientation_preds.device,
            dtype=orientation_preds.dtype
        )
        segment_vectors, segment_valid = segment_targets
        vectors, query_has_orientation = self._build_query_orientation_targets(
            orientation_preds=orientation_preds,
            segment_vectors=segment_vectors,
            segment_has_orientation=segment_valid,
            matching=matching
        )

        loss_dict = {}
        # only supervise orientations for tokens that have valid targets
        mask = query_has_orientation.bool()
        if mask.any():
            preds = orientation_preds[mask].reshape(-1, 2)
            target = vectors[mask].reshape(-1, 2)
            loss_outputs = self._orientation_loss((preds,), (target,))
            orientation_loss, _ = loss_outputs[0]
        else:
            orientation_loss = orientation_preds.new_tensor(0.0)
        weighted_orientation = orientation_loss * self._orientation_loss_weight
        loss_dict['token_orientation_loss_direction'] = weighted_orientation
        total_loss = weighted_orientation

        if self._supervise_confidence:
            # confidence is supervised for every token, as it indicates whether
            # the orientation of the token should be used in application.
            confidence_logits = predictions_post[
                'token_orientation_confidence_logits'
            ]
            target = query_has_orientation.float()
            confidence_loss = F.binary_cross_entropy_with_logits(
                confidence_logits,
                target
            )
            weighted_confidence = confidence_loss * self._confidence_loss_weight
            loss_dict['token_orientation_loss_confidence'] = weighted_confidence
            total_loss = total_loss + weighted_confidence

        loss_dict[self.mark_as_total('token_orientation')] = total_loss
        return loss_dict

    @append_profile_to_logs('token_orientation_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        return loss_dict, {}

    @append_profile_to_logs('token_orientation_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        orientation_preds = predictions_post['token_orientation_queries']
        matching = self.matching_cache[batch_idx]
        segment_targets = self._build_segment_orientation_targets(
            batch,
            device=orientation_preds.device,
            dtype=orientation_preds.dtype
        )
        segment_vectors, segment_valid = segment_targets
        targets, query_has_orientation = self._build_query_orientation_targets(
            orientation_preds=orientation_preds,
            segment_vectors=segment_vectors,
            segment_has_orientation=segment_valid,
            matching=matching
        )
        self._update_metrics(
            orientation_preds,
            targets,
            query_has_orientation
        )
        return loss_dict, {}

    def validation_epoch_end(self):
        logs = {}
        rad, deg = self._mae_metric.compute()
        logs['token_orientation_mae_rad'] = rad
        logs['token_orientation_mae_deg'] = deg
        self._mae_metric.reset()
        return {}, {}, logs
