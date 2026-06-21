# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import math
import warnings

import torch

# this module reuses Mask2Former's Hungarian matcher + per-point loss
# helpers from HuggingFace transformers. transformers is optional; if it
# is not installed we still want `from .mask import TokenMaskTaskHelper`
# to succeed so the parent package can import this module unconditionally.
# every public class then raises a helpful ImportError on instantiation.
IS_TRANSFORMERS_AVAILABLE = False
try:
    from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss
    from transformers.models.mask2former.modeling_mask2former import sample_point
    IS_TRANSFORMERS_AVAILABLE = True
except ImportError:    # pragma: no cover - no transformers installed
    warnings.warn(
        "Could not import transformers (HuggingFace). "
        "TokenMaskTaskHelper is disabled until "
        "`python -m pip install -e "
        "'./nicr-mt-scene-analysis[withtransformers]'` is run."
    )
    Mask2FormerHungarianMatcher = None
    Mask2FormerConfig = None
    Mask2FormerLoss = object  # placeholder base so subclass-def below works
    sample_point = None

from ...types import BatchType
from ...loss import BinaryCrossEntropyWithLogitsLoss
from ...loss import DiceLoss
from ..base import append_detached_losses_to_logs
from ..base import append_profile_to_logs
from .base import TokenMatchingCache
from .base import TokenMatchingTaskHelperBase


VALID_MASK_MODES = ['semantic', 'panoptic']


class _Mask2FormerLossWrapper(Mask2FormerLoss):
    def __init__(
        self,
        *,
        n_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float
    ):
        # The HF parent expects a config. We only reuse its helper methods, so
        # this minimal config carries the point-sampling parameters we need.
        config = Mask2FormerConfig(
            train_num_points=n_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio
        )
        super().__init__(config, weight_dict={})

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Mask2FormerLossWrapper proxies HF helpers; forward() is not "
            "expected to be called."
        )

    def loss_labels(self, *args, **kwargs):
        raise NotImplementedError(
            "Mask2FormerLossWrapper proxies HF helpers; loss_labels() is not "
            "expected to be called."
            "Please use a dedicated task helper for other losses."
        )

    def sample_points(self, logits: torch.Tensor) -> torch.Tensor:
        return self.sample_points_using_uncertainty(
            logits,
            self.calculate_uncertainty,
            self.num_points,  # Mask2FormerLoss already creates the member
            self.oversample_ratio,
            self.importance_sample_ratio
        )

    def select_matched_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: Sequence[torch.Tensor],
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(list(mask_labels))
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]
        return pred_masks, target_masks


class TokenMaskTaskHelper(TokenMatchingTaskHelperBase):
    def __init__(
        self,
        *,
        matching_cache: TokenMatchingCache,
        mask_mode: str = 'panoptic',
        n_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_scores_prefix: Optional[str] = None,
        class_labels_key: Optional[str] = None,
        class_label_offset: int = 0,
        class_weight: float = 0.0,
        unmatched_mask_coefficient: float = 0.0,
        unmatched_mask_topk_frac: float = 1.0
    ):
        if not IS_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "TokenMaskTaskHelper requires `transformers`. "
                "Install via "
                "`python -m pip install -e "
                "'./nicr-mt-scene-analysis[withtransformers]'`."
            )
        super().__init__(matching_cache=matching_cache)
        assert mask_mode in VALID_MASK_MODES, "mask_mode must be one of " \
            f"{VALID_MASK_MODES}, but got '{mask_mode}'."
        self._target_masks_key = f'{mask_mode}_token_masks'
        self._mask_coefficient = mask_coefficient
        self._dice_coefficient = dice_coefficient
        self._mask_loss_helper = _Mask2FormerLossWrapper(
            n_points=n_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio
        )
        self._loss_prefix = f'token_{mask_mode}'
        self._class_scores_prefix = class_scores_prefix
        self._class_labels_key = class_labels_key
        self._class_label_offset = class_label_offset
        self._class_weight = class_weight
        self._unmatched_mask_coefficient = unmatched_mask_coefficient
        assert 0.0 <= unmatched_mask_topk_frac <= 1.0, \
            "unmatched_mask_topk_frac must be in [0, 1]."
        self._unmatched_mask_topk_frac = unmatched_mask_topk_frac

    def initialize(self, device: torch.device):
        self._mask_loss_helper = self._mask_loss_helper.to(device)
        # matcher is created here and no in the Wrapper so its a little
        # cleaner if we need to modify it.
        self._matcher = Mask2FormerHungarianMatcher(
            cost_class=self._class_weight,
            cost_dice=self._dice_coefficient,
            cost_mask=self._mask_coefficient,
            num_points=self._mask_loss_helper.num_points,
        ).to(device)
        self._mask_loss = BinaryCrossEntropyWithLogitsLoss(
            reduction="none"
        ).to(device)
        self._dice_loss = DiceLoss().to(device)

    def compute_mask_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        mask_logits = self._collect_stage_tensors(
            predictions_post,
            key_prefix='token_mask'
        )
        class_labels = None
        if self._class_labels_key is not None:
            class_labels = batch[self._class_labels_key]
            if self._class_label_offset != 0:
                class_labels = tuple(
                    labels - self._class_label_offset
                    for labels in class_labels
                )

        # cast mask labels to float for loss computation.
        # this is required as GridSampler is only implemented for float tensors.
        mask_labels = tuple(
            entry.float() for entry in batch[self._target_masks_key]
        )
        # get number of masks in batch for normalization
        n_masks = sum(label.size(0) for label in mask_labels)
        # prepare keys for losses
        aux_count = len(mask_logits) - 1
        aux_keys = tuple(f'aux_{idx+1}' for idx in range(aux_count))
        keys = aux_keys + ('main',)

        # prepare loss dict
        loss_dict = {}
        mask_losses = []
        dice_losses = []
        unmatched_mask_losses = []
        if self._class_labels_key is None:
            class_labels = tuple(
                torch.zeros(
                    label.size(0),
                    device=mask_logits[0].device,
                    dtype=torch.long
                )
                for label in mask_labels
            )
        class_scores = None
        if self._class_scores_prefix is not None:
            class_scores = self._collect_stage_tensors(
                predictions_post,
                key_prefix=self._class_scores_prefix
            )
        else:
            class_scores = tuple(
                torch.zeros(
                    mask_stage.shape[0],
                    mask_stage.shape[1],
                    1,
                    device=mask_stage.device,
                    dtype=mask_stage.dtype
                )
                for mask_stage in mask_logits
            )
        # compute losses for each stage
        for key, stage_mask_logits, stage_class_scores in zip(
            keys,
            mask_logits,
            class_scores
        ):
            indices = self._matcher(
                masks_queries_logits=stage_mask_logits,
                mask_labels=mask_labels,
                class_queries_logits=stage_class_scores,
                class_labels=class_labels
            )
            self.matching_cache.set_stage(batch_idx, key, list(indices))
            stage_mask, stage_dice = self._compute_stage_losses(
                pred_logits=stage_mask_logits,
                pred_match_indices=indices,
                mask_labels=mask_labels,
                n_masks=n_masks
            )
            loss_dict[f'{self._loss_prefix}_loss_mask_{key}'] = stage_mask
            loss_dict[f'{self._loss_prefix}_loss_dice_{key}'] = stage_dice
            mask_losses.append(stage_mask)
            dice_losses.append(stage_dice)
            if self._unmatched_mask_coefficient > 0.0:
                stage_unmatched = self._compute_unmatched_mask_loss(
                    pred_logits=stage_mask_logits,
                    pred_match_indices=indices
                )
                loss_dict[
                    f'{self._loss_prefix}_loss_unmatched_{key}'
                ] = stage_unmatched
                unmatched_mask_losses.append(stage_unmatched)
        # sum up all stage losses.
        # we dont normalize here on purpose (same as in eomt).
        mask_total = torch.sum(torch.stack(mask_losses))
        dice_total = torch.sum(torch.stack(dice_losses))
        unmatched_total = None
        if unmatched_mask_losses:
            unmatched_total = torch.sum(torch.stack(unmatched_mask_losses))
        loss_dict[f'{self._loss_prefix}_loss_mask'] = mask_total
        loss_dict[f'{self._loss_prefix}_loss_dice'] = dice_total
        if unmatched_total is not None:
            loss_dict[f'{self._loss_prefix}_loss_unmatched'] = unmatched_total
        total = mask_total + dice_total
        if unmatched_total is not None:
            total = total + unmatched_total
        loss_dict[self.mark_as_total(f'{self._loss_prefix}_mask')] = total
        return loss_dict

    def _collect_stage_tensors(
        self,
        predictions_post: BatchType,
        *,
        key_prefix: str
    ) -> Tuple[torch.Tensor, ...]:
        side_key = f'{key_prefix}_side_outputs'
        main_key = f'{key_prefix}_output'
        side_outputs = predictions_post[side_key]
        main = predictions_post[main_key]
        return side_outputs + (main,)

    def _compute_stage_losses(
        self,
        *,
        pred_logits: torch.Tensor,
        pred_match_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        mask_labels: Sequence[torch.Tensor],
        n_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # select K masks based on the matching indices from the matcher
        # which match Q queries to ground truth masks.
        # pred_logits [B, Q, H, W] -> [K, 1, H, W]
        pred_masks, target_masks = self._mask_loss_helper.select_matched_masks(
            pred_logits,
            mask_labels,
            pred_match_indices
        )
        # sample n_points per predicted mask based on uncertainty
        point_coordinates = self._mask_loss_helper.sample_points(pred_masks)
        # select the mask values at the sampled point coordinates
        # and drop the singleton dimension so we have [K, n_points]
        point_labels = sample_point(
            target_masks,
            point_coordinates,
            align_corners=False,
        ).squeeze(1)
        # get the predicted logits at the sampled point coordinates
        # and drop the singleton dimension so we have [K, n_points]
        point_logits = sample_point(
            pred_masks,
            point_coordinates,
            align_corners=False,
        ).squeeze(1)
        # compute bce on the raw logits as loss function operates on logits
        bce_loss, _ = self._mask_loss(
            input_tensors=(point_logits,),
            target_tensors=(point_labels,)
        )[0]
        # normalize by number of masks to keep loss scale consistent
        stage_mask = bce_loss.mean(1).sum() / n_masks
        # compute dice loss on the probabilities after sigmoid. sigmoid is
        # applied here on purpose, not in the postprocessing step, to reduce
        # memory consumption.
        point_probs = point_logits.sigmoid()
        dice_sum, _ = self._dice_loss(
            input_tensors=(point_probs,),
            target_tensors=(point_labels,)
        )[0]
        # normalize by number of masks to keep loss scale consistent
        stage_dice = dice_sum / n_masks
        return (
            stage_mask * self._mask_coefficient,
            stage_dice * self._dice_coefficient
        )

    def _compute_unmatched_mask_loss(
        self,
        *,
        pred_logits: torch.Tensor,
        pred_match_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        if self._unmatched_mask_topk_frac <= 0.0:
            return pred_logits.sum() * 0.0
        batch_size, n_queries = pred_logits.shape[:2]
        unmatched_masks = []
        for sample_idx, (src_indices, _) in enumerate(pred_match_indices):
            if src_indices.numel() == 0:
                unmatched_idx = torch.arange(
                    n_queries,
                    device=pred_logits.device
                )
            else:
                matched = torch.zeros(
                    n_queries,
                    device=pred_logits.device,
                    dtype=torch.bool
                )
                matched[src_indices] = True
                unmatched_idx = (~matched).nonzero(as_tuple=False).squeeze(1)
            if unmatched_idx.numel() == 0:
                continue
            if self._unmatched_mask_topk_frac < 1.0:
                scores = pred_logits[sample_idx].amax(dim=(-2, -1))
                n_unmatched = unmatched_idx.numel()
                topk = max(
                    1,
                    int(math.ceil(self._unmatched_mask_topk_frac * n_unmatched))
                )
                if topk < unmatched_idx.numel():
                    _, topk_pos = scores[unmatched_idx].topk(topk)
                    unmatched_idx = unmatched_idx[topk_pos]
            unmatched_masks.append(pred_logits[sample_idx, unmatched_idx])
        if not unmatched_masks:
            return pred_logits.sum() * 0.0
        pred_masks = torch.cat(unmatched_masks, dim=0)[:, None]
        point_coordinates = self._mask_loss_helper.sample_points(pred_masks)
        point_logits = sample_point(
            pred_masks,
            point_coordinates,
            align_corners=False,
        ).squeeze(1)
        target = torch.zeros_like(point_logits)
        bce_loss, _ = self._mask_loss(
            input_tensors=(point_logits,),
            target_tensors=(target,)
        )[0]
        stage_unmatched = bce_loss.mean(1).mean()
        return stage_unmatched * self._unmatched_mask_coefficient


    @append_profile_to_logs('token_mask_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self.compute_mask_losses(batch, batch_idx, predictions_post)
        return loss_dict, {}

    @append_profile_to_logs('token_mask_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self.compute_mask_losses(batch, batch_idx, predictions_post)
        return loss_dict, {}

    def validation_epoch_end(self):
        return {}, {}, {}
