# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Type

from ....types import DecoderRawOutputType
from ....types import EncoderSkipsType
from ....types import TokenDecoderInputType
from ...postprocessing import get_postprocessing_class
from ...postprocessing import PostProcessingType
from .base import TokenLinearHeadDecoder


class TokenSceneDecoder(TokenLinearHeadDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        modality: Optional[str] = None,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class(
            'scene'
        )
    ):
        super().__init__(embed_dim=embed_dim,
                         output_dim=n_classes,
                         modality=modality,
                         postprocessing=postprocessing,
                         use_cls_token=True,
                         squeeze_queries=True)

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[Dict[str, Any]] = None
    ) -> DecoderRawOutputType:
        # scene classification operates on the CLS token only and does not need
        # side outputs.
        _, queries = self._resolve_queries(x)
        logits = self._compute_head_output(queries)
        return logits, None
