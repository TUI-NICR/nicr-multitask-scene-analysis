# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Union

from .normal import NormalTaskHelper
from .scene import SceneTaskHelper
from .semantic import SemanticTaskHelper
from .dense_visual_embedding import DenseVisualEmbeddingTaskHelper
from .instance import InstanceTaskHelper
from .panoptic import PanopticTaskHelper
from .token_based import IS_TRANSFORMERS_AVAILABLE
from .token_based import TokenMaskTaskHelper
from .token_based import TokenMatchingCache
from .token_based import TokenSemanticTaskHelper
from .token_based import TokenPanopticTaskHelper
from .token_based import TokenOrientationTaskHelper
from .token_based import TokenEmbeddingTaskHelper
from .token_based import TokenSceneTaskHelper

TokenTaskHelperType = Union[
    TokenMaskTaskHelper,
    TokenSemanticTaskHelper,
    TokenPanopticTaskHelper,
    TokenOrientationTaskHelper,
    TokenEmbeddingTaskHelper,
    TokenSceneTaskHelper,
]

TaskHelperType = Union[NormalTaskHelper, SceneTaskHelper,
                       SemanticTaskHelper, DenseVisualEmbeddingTaskHelper,
                       InstanceTaskHelper, PanopticTaskHelper,
                       TokenTaskHelperType]
