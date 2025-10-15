# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...data.preprocessing.resize import get_fullres_key
from ...data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .dense_base import DensePostprocessingBase


class DenseVisualEmbeddingPostprocessing(DensePostprocessingBase):

    def __init__(
        self,
        with_text_embeddings_per_class: bool = False,
        text_embeddings_per_class: Optional[torch.Tensor] = None,
        with_mean_visual_embedding_per_class: bool = False,
        mean_visual_embedding_per_class: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__()
        self.with_semantic_text_embeddings = with_text_embeddings_per_class
        self._semantic_text_embeddings = None
        if self.with_semantic_text_embeddings:
            assert text_embeddings_per_class is not None
            self._semantic_text_embeddings = text_embeddings_per_class
            self._semantic_text_embeddings = self._semantic_text_embeddings.unsqueeze(-1).unsqueeze(-1)

        self.with_mean_visual_embedding_per_class = with_mean_visual_embedding_per_class
        self._mean_visual_embedding_per_class = None
        if self.with_mean_visual_embedding_per_class:
            assert mean_visual_embedding_per_class is not None
            self._mean_visual_embedding_per_class = mean_visual_embedding_per_class
            self._mean_visual_embedding_per_class = self._mean_visual_embedding_per_class.unsqueeze(-1).unsqueeze(-1)

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {
            'dense_visual_embedding_output': output,
            'dense_visual_embedding_side_outputs': side_outputs
        }

        return r_dict

    def _add_semantic_results_to_dict(
        self,
        output: torch.Tensor,
        weight: torch.Tensor,
        crop_slices: Tuple[slice, slice],
        resize_shape: Tuple[int, int],
        r_dict: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor]]],
        output_key: str,
        softmax_scores_key: str,
        score_key: str,
        idx_key: str,
        fullres_output_key: str,
        fullres_softmax_scores_key: str,
        fullres_score_key: str,
        fullres_idx_key: str
    ) -> PostprocessingOutputType:
        # Compute the cosine similarity
        # -> conv2d with semantic text embeddings as weights
        # -> this is equivalent to a dot product between the output and the
        #    semantic text embeddings, because the output and the semantic
        #    text embeddings are normalized.
        semantic_logits = F.conv2d(input=output, weight=weight)

        # apply softmax and get max / argmax
        semantic_pred = F.softmax(semantic_logits, dim=1)
        semantic_score, semantic_idx = torch.max(semantic_pred, dim=1)

        # write results to dict
        r_dict.update({
            output_key: semantic_logits,
            softmax_scores_key: semantic_pred,
            score_key: semantic_score,
            idx_key: semantic_idx,
        })

        # get fullres outputs
        semantic_logits_fullres = self._crop_to_valid_region_and_resize_prediction(
            semantic_logits, valid_region_slices=crop_slices, shape=resize_shape, mode='bilinear')

        semantic_pred_fullres = F.softmax(semantic_logits_fullres, dim=1)
        semantic_score_fullres, semantic_idx_fullres = torch.max(semantic_pred_fullres, dim=1)

        # write fullres results to dict
        r_dict.update({
            fullres_output_key: semantic_logits_fullres,
            fullres_softmax_scores_key: semantic_pred_fullres,
            fullres_score_key: semantic_score_fullres,
            fullres_idx_key: semantic_idx_fullres,
        })
        return r_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {
            'dense_visual_embedding_output': output,
            'dense_visual_embedding_side_outputs': side_outputs
        }

        # normalize output anlong channel dim
        output /= output.norm(dim=1, keepdim=True)

        # crop and resize output to full resolution (original shape)
        crop_slices, resize_shape = get_valid_region_slices_and_fullres_shape(
            batch, 'semantic'
        )

        if self.with_semantic_text_embeddings:
            r_dict = self._add_semantic_results_to_dict(
                output=output,
                weight=self._semantic_text_embeddings,
                crop_slices=crop_slices,
                resize_shape=resize_shape,
                r_dict=r_dict,
                output_key='dense_visual_embedding_text_based_semantic_output',
                softmax_scores_key='dense_visual_embedding_text_based_semantic_softmax_scores',
                score_key='dense_visual_embedding_text_based_semantic_score',
                idx_key='dense_visual_embedding_text_based_semantic_idx',
                fullres_output_key=get_fullres_key('dense_visual_embedding_text_based_semantic_output'),
                fullres_softmax_scores_key=get_fullres_key('dense_visual_embedding_text_based_semantic_softmax_scores'),
                fullres_score_key=get_fullres_key('dense_visual_embedding_text_based_semantic_score'),
                fullres_idx_key=get_fullres_key('dense_visual_embedding_text_based_semantic_idx'),
            )

        if self.with_mean_visual_embedding_per_class:
            r_dict = self._add_semantic_results_to_dict(
                output=output,
                weight=self._mean_visual_embedding_per_class,
                crop_slices=crop_slices,
                resize_shape=resize_shape,
                r_dict=r_dict,
                output_key='dense_visual_embedding_visual_mean_based_semantic_output',
                softmax_scores_key='dense_visual_embedding_visual_mean_based_semantic_softmax_scores',
                score_key='dense_visual_embedding_visual_mean_based_semantic_score',
                idx_key='dense_visual_embedding_visual_mean_based_semantic_idx',
                fullres_output_key=get_fullres_key('dense_visual_embedding_visual_mean_based_semantic_output'),
                fullres_softmax_scores_key=get_fullres_key('dense_visual_embedding_visual_mean_based_semantic_softmax_scores'),
                fullres_score_key=get_fullres_key('dense_visual_embedding_visual_mean_based_semantic_score'),
                fullres_idx_key=get_fullres_key('dense_visual_embedding_visual_mean_based_semantic_idx'),
            )

        return r_dict
