# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch

from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from .base import TokenPostprocessingBase


class TokenMaskPostprocessing(TokenPostprocessingBase):
    def __init__(self):
        super().__init__(output_key='token_mask_output')

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        output, side_outputs = data
        side_outputs = tuple(side_outputs or ())
        out_dict.update({
            'token_mask_output': output,
            'token_mask_side_outputs': side_outputs,
        })
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
        out_dict['token_mask_probs'] = torch.sigmoid(
            out_dict['token_mask_output']
        )
        return out_dict
