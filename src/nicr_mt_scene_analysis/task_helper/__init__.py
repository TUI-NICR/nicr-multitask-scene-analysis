# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Union

from .normal import NormalTaskHelper
from .scene import SceneTaskHelper
from .semantic import SemanticTaskHelper
from .dense_visual_embedding import DenseVisualEmbeddingTaskHelper
from .instance import InstanceTaskHelper
from .panoptic import PanopticTaskHelper

TaskHelperType = Union[NormalTaskHelper, SceneTaskHelper,
                       SemanticTaskHelper, DenseVisualEmbeddingTaskHelper,
                       InstanceTaskHelper, PanopticTaskHelper]
