# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional

import torch

from ....data.preprocessing.resize import get_fullres_key
from ....data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from ..dense_base import DensePostprocessingBase
from .base import TokenPostprocessingBase


def add_dense_semantic_outputs(
    *,
    dense_postprocessor: DensePostprocessingBase,
    prefix: str,
    dense_scores: torch.Tensor,
    dense_score_max: torch.Tensor,
    dense_labels: torch.Tensor,
    batch: BatchType,
    out_dict: PostprocessingOutputType,
    fullres_mode: str = 'bilinear'
) -> None:
    # write semantic-style tensors into the shared out_dict at the network's
    # working resolution and resized to the full image size used by metrics.
    # The key suffixes are output, softmax_scores, score, and idx. This helper
    # lives next to TokenSemanticPostprocessing because the key naming and the
    # 'semantic' crop reference are specific to that output set.
    out_dict.update({
        f'{prefix}_output': dense_scores,
        f'{prefix}_softmax_scores': dense_scores,
        f'{prefix}_score': dense_score_max,
        f'{prefix}_idx': dense_labels,
    })
    crop_slices, resize_shape = get_valid_region_slices_and_fullres_shape(
        batch, 'semantic'
    )
    crop_and_resize = (
        dense_postprocessor._crop_to_valid_region_and_resize_prediction
    )
    dense_logits_fullres = crop_and_resize(
        dense_scores,
        valid_region_slices=crop_slices,
        shape=resize_shape,
        mode=fullres_mode
    )
    dense_scores_fullres, dense_labels_fullres = torch.max(
        dense_logits_fullres, dim=1
    )
    out_dict.update({
        get_fullres_key(f'{prefix}_output'): dense_logits_fullres,
        get_fullres_key(f'{prefix}_softmax_scores'): dense_logits_fullres,
        get_fullres_key(f'{prefix}_score'): dense_scores_fullres,
        get_fullres_key(f'{prefix}_idx'): dense_labels_fullres,
    })


class TokenSemanticPostprocessing(
    TokenPostprocessingBase,
    DensePostprocessingBase
):
    def __init__(self, *, n_classes: Optional[int] = None):
        super().__init__(output_key='token_semantic_output')
        self._n_classes = n_classes

    def _clip_class_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._n_classes is None:
            return tensor
        if tensor.shape[-1] <= self._n_classes:
            return tensor
        return tensor[..., :self._n_classes]

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        logits, side_semantic = data
        out_dict['token_semantic_output'] = logits
        out_dict['token_semantic_side_outputs'] = tuple(side_semantic)
        return out_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        out_dict = self._postprocess_training(
            data, batch, out_dict=out_dict
        )
        logits = out_dict['token_semantic_output']
        mask_probs = out_dict['token_mask_probs']
        probs = torch.softmax(logits, dim=-1)
        probs = self._clip_class_channels(probs)
        token_scores, token_labels = torch.max(probs, dim=-1)
        out_dict.update({
            'token_semantic_softmax_scores': probs,
            'token_semantic_segmentation_score': token_scores,
            'token_semantic_segmentation_idx': token_labels,
        })

        dense_scores = torch.einsum('bqhw,bqc->bchw', mask_probs, probs)
        dense_scores_max, dense_labels = torch.max(dense_scores, dim=1)
        # dense per-pixel predictions derived from the token masks
        add_dense_semantic_outputs(
            dense_postprocessor=self,
            prefix='token_semantic_dense',
            dense_scores=dense_scores,
            dense_score_max=dense_scores_max,
            dense_labels=dense_labels,
            batch=batch,
            out_dict=out_dict,
            fullres_mode='bilinear'
        )
        return out_dict
