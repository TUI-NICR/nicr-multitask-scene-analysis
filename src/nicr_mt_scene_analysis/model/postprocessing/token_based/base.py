# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional

import abc

from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from ..base import PostprocessingBase


class TokenPostprocessingBase(PostprocessingBase):

    def __init__(self, output_key: str):
        super().__init__()
        self._output_key = output_key

    def postprocess(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        is_training: bool = True,
        out_dict: Optional[PostprocessingOutputType] = None
    ) -> PostprocessingOutputType:
        # override of PostprocessingBase.postprocess that threads out_dict
        # so chained token postprocessors can share one output dict.
        if out_dict is None:
            out_dict = {}
        if is_training:
            return self._postprocess_training(data, batch, out_dict=out_dict)
        return self._postprocess_inference(data, batch, out_dict=out_dict)

    @abc.abstractmethod
    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        pass

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        # default behavior: same as in training
        return self._postprocess_training(data, batch, out_dict=out_dict)
