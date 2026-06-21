# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch

from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from ....utils import biternion2rad
from .base import TokenPostprocessingBase


class TokenOrientationPostprocessing(TokenPostprocessingBase):
    def __init__(self, *, with_confidence: bool = False):
        # stores the normalized biternion (cos, sin) emitted per query by
        # TokenOrientationDecoder.
        super().__init__(output_key='token_orientation_queries')
        self._with_confidence = with_confidence

    def _split_outputs(self, data: DecoderRawOutputType):
        outputs, side_outputs = data
        assert side_outputs is None, \
            'Token orientation decoders do not provide side outputs.'
        if self._with_confidence:
            # decoder guarantees this tuple layout
            orientations, confidence_logits = outputs
        else:
            # base decoder already validated tensor output
            orientations = outputs
            confidence_logits = None
        return orientations, confidence_logits

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        orientations, confidence_logits = self._split_outputs(data)
        result = {'token_orientation_queries': orientations}
        if self._with_confidence:
            result['token_orientation_confidence_logits'] = confidence_logits
        out_dict.update(result)
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
        orientations = out_dict['token_orientation_queries']
        assert orientations.shape[-1] == 2
        batch_size, n_queries, _ = orientations.shape
        angles = biternion2rad(
            orientations.reshape(batch_size * n_queries, 2)
        ).view(batch_size, n_queries)
        out_dict['token_orientation_angles'] = angles
        if self._with_confidence:
            logits = out_dict['token_orientation_confidence_logits']
            confidence_probs = torch.sigmoid(logits)
            out_dict['token_orientation_confidence_probs'] = confidence_probs
        return out_dict
