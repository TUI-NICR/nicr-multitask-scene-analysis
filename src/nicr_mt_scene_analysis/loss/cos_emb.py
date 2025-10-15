# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple

import torch

from .base import LossBase


class CosineEmbeddingLoss(LossBase):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        # reduce later, so we can get cos loss per element
        self._loss = torch.nn.CosineEmbeddingLoss(reduction='none')
        self._reduction = reduction
        assert self._reduction in ('sum', 'mean', 'none')

    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor,
        target_similarity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:

        # Pytorch computes the loss between two inputs (input_ and target)
        # and requires a target label for each pair of inputs to indicate
        # if the pair should be considered similar (label = 1)
        # or unsimilar (label = -1).
        # If no target_similarity is given, we assume that each pair of inputs
        # should be considered similar (label = 1).
        if target_similarity is None:
            # if target_labels is not given assume that each entry in 'targets' corresponds
            # to one entry in 'input_' -> simply pass ones() as labels to CosineEmbeddingLoss
            target_similarity = torch.ones(target.shape[0]).to(target.device)

        # Compute the loss
        loss = self._loss(input_, target, target_similarity)

        # Apply reduction
        if 'sum' == self._reduction:
            # first average over channel/feature axis to get the mse per element
            if loss.ndim in (2, 4):
                # bchw/bn: we have a channel/feature axis at index 1
                loss = torch.mean(loss, dim=1)
            n_elements = loss.numel()    # number of pixels
            loss = loss.sum()
        elif 'mean' == self._reduction:
            loss = loss.mean()
            n_elements = 1
        else:
            n_elements = input_.numel()

        return loss, n_elements
