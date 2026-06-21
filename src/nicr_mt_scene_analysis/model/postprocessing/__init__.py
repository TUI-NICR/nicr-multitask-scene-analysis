# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Type, Union

from ...utils import partial_class
from .instance import InstancePostprocessing
from .normal import NormalPostprocessing
from .panoptic import PanopticPostprocessing
from .scene import ScenePostprocessing
from .semantic import SemanticPostprocessing
from .dense_visual_embedding import DenseVisualEmbeddingPostprocessing
from .token_based import TokenMaskPostprocessing
from .token_based import TokenSemanticPostprocessing
from .token_based import TokenOrientationPostprocessing
from .token_based import TokenPanopticPostprocessing
from .token_based import TokenVisualEmbeddingPostprocessing
from .token_based import TokenImageEmbeddingPostprocessing


PostProcessingType = Type[Union[InstancePostprocessing,
                                NormalPostprocessing,
                                PanopticPostprocessing,
                                ScenePostprocessing,
                                SemanticPostprocessing,
                                DenseVisualEmbeddingPostprocessing,
                                TokenMaskPostprocessing,
                                TokenSemanticPostprocessing,
                                TokenOrientationPostprocessing,
                                TokenPanopticPostprocessing,
                                TokenVisualEmbeddingPostprocessing,
                                TokenImageEmbeddingPostprocessing]]


def get_postprocessing_class(
    name: str,
    **kwargs: Any
) -> Type[PostProcessingType]:

    if 'semantic' == name:
        cls = SemanticPostprocessing
    elif 'dense-visual-embedding' == name:
        cls = DenseVisualEmbeddingPostprocessing
    elif 'instance' == name:
        cls = InstancePostprocessing
    elif 'panoptic' == name:
        cls = PanopticPostprocessing
    elif 'normal' == name:
        cls = NormalPostprocessing
    elif 'scene' == name:
        cls = ScenePostprocessing
    elif 'token-mask' == name:
        cls = TokenMaskPostprocessing
    elif 'token-semantic' == name:
        cls = TokenSemanticPostprocessing
    elif 'token-orientation' == name:
        cls = TokenOrientationPostprocessing
    elif 'token-panoptic' == name:
        cls = TokenPanopticPostprocessing
    elif 'token-visual-embedding' == name:
        cls = TokenVisualEmbeddingPostprocessing
    elif 'token-image-embedding' == name:
        cls = TokenImageEmbeddingPostprocessing
    else:
        raise ValueError(f"Unknown postprocessing: '{name}'")

    return partial_class(cls, **kwargs)
