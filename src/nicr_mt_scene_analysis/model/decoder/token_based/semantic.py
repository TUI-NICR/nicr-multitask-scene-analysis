# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Type

from ...postprocessing import get_postprocessing_class
from ...postprocessing import PostProcessingType
from .base import TokenLinearHeadDecoder


class TokenSemanticDecoder(TokenLinearHeadDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        modality: Optional[str] = None,
        postprocessing: Optional[Type[PostProcessingType]] = None,
        side_output_stage_indices: Optional[Tuple[int, ...]] = None
    ):
        if postprocessing is None:
            postprocessing = get_postprocessing_class(
                'token-semantic',
                n_classes=n_classes
            )
        super().__init__(embed_dim=embed_dim,
                         output_dim=n_classes,
                         modality=modality,
                         side_output_stage_indices=side_output_stage_indices,
                         postprocessing=postprocessing)
