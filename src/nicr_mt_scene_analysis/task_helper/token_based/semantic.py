# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ...data.preprocessing.resize import get_fullres
from ...data.preprocessing.resize import get_fullres_key
from ...loss import CrossEntropyLossSemantic
from ...metric import MeanIntersectionOverUnion
from ...types import BatchType
from ...visualization import visualize_heatmap_pil
from ...visualization import visualize_semantic_pil
from ..base import append_detached_losses_to_logs
from ..base import append_profile_to_logs
from .base import TokenMatchingCache
from .base import TokenMatchingTaskHelperBase


class TokenSemanticTaskHelper(TokenMatchingTaskHelperBase):
    def __init__(
        self,
        *,
        n_classes: int,
        matching_cache: TokenMatchingCache,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        examples_cmap: Optional[Sequence[Tuple[int, int, int]]] = None,
        semantic_coefficient: float = 4.0
    ):
        super().__init__(matching_cache=matching_cache)
        self._n_classes = n_classes
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._semantic_coefficient = semantic_coefficient
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
            # Dataset weights may still include void. The token semantic head
            # only predicts non-void classes, so drop the first weight here.
            if weights.shape[0] == self._n_classes + 1:
                weights = weights[1:]
            elif weights.shape[0] != self._n_classes:
                raise ValueError(
                    f"Expected {self._n_classes} semantic weights "
                    f"(without void) but received {weights.shape[0]}."
                )
        self._weighted_reduction = weights is not None
        self._loss = CrossEntropyLossSemantic(
            weights=weights,
            label_smoothing=self._label_smoothing,
            weighted_reduction=self._weighted_reduction
        )
        self._metric_iou = MeanIntersectionOverUnion(
            n_classes=self._n_classes
        )
        self._metric_iou.reset()

    def _build_targets(
        self,
        *,
        logits: torch.Tensor,
        class_labels: Sequence[torch.Tensor],
        matching: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        # logits: [B, Q, C] with Q query tokens and C semantic classes
        batch_size, n_queries = logits.shape[:2]
        # targets: [B, Q], semantic labels, 0 for unmatched/ignored queries.
        # CrossEntropyLossSemantic shifts labels by -1 internally, turning 0
        # into the ignore_index (-1) expected by CE.
        targets = logits.new_full(
            (batch_size, n_queries),
            fill_value=0,
            dtype=torch.long
        )
        for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
            # matching links predicted query indices to target mask indices
            labels = class_labels[sample_idx]
            targets[sample_idx, src_indices] = labels[tgt_indices]
        return targets

    def _compute_cross_entropy(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # No matched semantic targets in this stage. Return a zero loss on the
        # right device and keep the graph connected to the logits. This can
        # happen for unlucky ground-truth resizes + crops (e.g. Hypersim)
        if (targets > 0).sum() == 0:
            return logits.sum() * 0.0
        # CrossEntropyLossSemantic expects the class dimension at index 1, so
        # transpose token logits from [B, Q, C] to [B, C, Q].
        loss, n_elements = self._loss(
            input_tensors=(logits.transpose(1, 2),),
            target_tensors=(targets,)
        )[0]
        # Unweighted CE returns a summed loss. With class weights, the loss
        # object already returns the weighted mean used by torch CE.
        if not self._weighted_reduction:
            loss = loss / n_elements
        return loss

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        output = predictions_post['token_semantic_output']
        side_outputs = predictions_post['token_semantic_side_outputs']
        stage_logits = side_outputs + (output,)
        aux_keys = tuple(f'aux_{idx+1}' for idx in range(len(side_outputs)))
        keys = aux_keys + ('main',)
        class_labels = []
        for label in batch['semantic_token_labels']:
            mask = label != 0
            class_labels.append(label[mask])
        loss_dict = {}
        # prepare memory for total loss computation
        total_loss = output.sum() * 0.0 # Create zero tensor on correct device
        # iterate over all stages (side outputs + main output)
        for key, pred_logits in zip(keys, stage_logits):
            # reuse matching from cache which was computed during mask loss.
            # matching will contain tuples of (pred_indices, target_indices)
            matching = self.matching_cache.get_stage(batch_idx, key)
            targets = self._build_targets(
                logits=pred_logits,
                class_labels=class_labels,
                matching=matching
            )
            loss = self._compute_cross_entropy(
                logits=pred_logits,
                targets=targets
            )
            loss_dict[f'token_semantic_loss_{key}'] = loss
            total_loss = total_loss + loss
        total_loss = total_loss * self._semantic_coefficient
        loss_dict['token_semantic_loss'] = total_loss
        loss_dict[self.mark_as_total('token_semantic')] = total_loss
        return loss_dict

    @append_profile_to_logs('token_semantic_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        return loss_dict, {}

    @append_profile_to_logs('token_semantic_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(batch, batch_idx, predictions_post)
        target_full = get_fullres(batch, 'semantic').long()
        preds_full = predictions_post[
            get_fullres_key('token_semantic_dense_idx')
        ].long()
        mask = target_full != 0    # mask of non-void pixels
        preds = preds_full[mask]
        target = target_full[mask] - 1    # first apply mask -> -1 is safe
        metric_device = self._metric_iou.confmat.device
        self._metric_iou.update(
            preds=preds.to(metric_device),
            target=target.to(metric_device)
        )

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            dense_idx = predictions_post['token_semantic_dense_idx']
            if dense_idx is not None and self._examples_cmap is not None:
                example = dense_idx[0].detach().cpu().numpy()
                key = f'token_semantic_example_batch_idx_{batch_idx}_0'
                self._examples[key] = visualize_semantic_pil(
                    semantic_img=example,
                    colors=self._examples_cmap
                )
            dense_score = predictions_post['token_semantic_dense_score']
            if dense_score is not None:
                example = dense_score[0].detach().cpu().numpy()
                key = f'token_semantic_example_batch_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=example,
                    min_=0,
                    max_=1
                )
        return loss_dict, {}

    def validation_epoch_end(self):
        artifacts, examples, logs = {}, {}, {}
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs['token_semantic_miou'] = miou.cpu()
        artifacts['token_semantic_ious_per_class'] = ious.clone().cpu()
        self._metric_iou.reset()
        examples = dict(self._examples)
        self._examples = {}
        return artifacts, examples, logs
