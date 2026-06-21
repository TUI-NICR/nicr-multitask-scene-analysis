# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

from torch import Tensor

from .base import LossBase


class DiceLoss(LossBase):

    def _compute_loss(
        self,
        input_: Tensor,
        target: Tensor
    ) -> Tuple[Tensor, int]:
        probs = input_.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        numerator = 2 * (probs * target_flat).sum(dim=-1)
        denominator = probs.sum(dim=-1) + target_flat.sum(dim=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum(), int(loss.numel())
