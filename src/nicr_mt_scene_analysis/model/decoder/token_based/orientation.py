# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Type

from torch import nn

from ....utils import OrientationOutputNormalization
from ....types import DecoderRawOutputType
from ....types import EncoderSkipsType
from ....types import TokenDecoderInputType
from ...postprocessing import get_postprocessing_class
from ...postprocessing import PostProcessingType
from .base import TokenLinearHeadDecoder


class TokenOrientationDecoder(TokenLinearHeadDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        modality: Optional[str] = None,
        predict_confidence: bool = False,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class(
            'token-orientation'
        )
    ):
        # optional per-token confidence head for self-gating against
        # targets that may lack orientation labels.
        self._predict_confidence = predict_confidence
        if predict_confidence:
            postprocessing = get_postprocessing_class(
                'token-orientation',
                with_confidence=predict_confidence
            )
        super().__init__(embed_dim=embed_dim,
                         output_dim=2,
                         modality=modality,
                         # output a normalized biternion (cos, sin) so each
                         # query encodes a unit vector as done in the dense
                         # head.
                         output_transform=(
                             OrientationOutputNormalization(dim=-1)
                         ),
                         postprocessing=postprocessing)
        if self._predict_confidence:
            self._confidence_head = nn.Linear(embed_dim, 1)

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> DecoderRawOutputType:
        _, queries = self._resolve_queries(x)
        orientations = self._compute_head_output(queries)
        if not self._predict_confidence:
            return orientations, None
        # confidence logits indicate whether the token should be used for
        # orientation supervision/inference. squeezing keeps shape [B, Q].
        confidence_logits = self._confidence_head(queries).squeeze(-1)
        return (orientations, confidence_logits), None
