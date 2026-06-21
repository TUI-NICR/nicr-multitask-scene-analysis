# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Type

from ...postprocessing import PostProcessingType
from .semantic import TokenSemanticDecoder


class TokenPanopticDecoder(TokenSemanticDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        modality: Optional[str] = None,
        postprocessing: Optional[Type[PostProcessingType]] = None,
        side_output_stage_indices: Optional[Tuple[int, ...]] = None
    ):
        super().__init__(
            embed_dim=embed_dim,
            n_classes=n_classes,
            modality=modality,
            postprocessing=postprocessing,
            side_output_stage_indices=side_output_stage_indices
        )
