# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ...types import BatchType
from ..scene import SceneTaskHelper
from .base import TokenTaskHelperBase


class TokenSceneTaskHelper(SceneTaskHelper, TokenTaskHelperBase):
    def __init__(
        self,
        *,
        n_classes: int,
        class_weights: Optional[np.ndarray] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__(
            n_classes=n_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing
        )

    def _collect_stage_logits(
        self,
        predictions_post: BatchType
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        logits = predictions_post['scene_output']
        # NOTE: TokenSceneDecoder does not provide side outputs, so we never
        # return them.
        return logits, ()

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        logits, side_outputs = self._collect_stage_logits(predictions_post)
        targets = batch['scene'].long() - 1
        tensors = (logits,) + side_outputs
        loss_dict = {}
        stage_losses = []
        for idx, tensor in enumerate(tensors):
            loss = self._loss(input=tensor, target=targets)
            key = 'main' if idx == 0 else f'aux_{idx}'
            loss_dict[f'token_scene_loss_{key}'] = loss
            stage_losses.append(loss)
        total_loss = sum(stage_losses) / len(stage_losses)
        loss_dict[self.mark_as_total('token_scene')] = total_loss
        return loss_dict
