# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import LossBase


class BinaryCrossEntropyWithLogitsLoss(LossBase):
    def __init__(
        self,
        *,
        pos_weight: Optional[Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )

    def _compute_loss(
        self,
        input_: Tensor,
        target: Tensor
    ) -> Tuple[Tensor, int]:
        loss = self._loss(input_, target)
        n_elements = target.numel()
        return loss, n_elements
