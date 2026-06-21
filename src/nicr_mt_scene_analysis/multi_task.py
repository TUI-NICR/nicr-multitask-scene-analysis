# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""

KNOWN_TASKS = (
    'semantic',    # semantic segmentation
    'dense-visual-embedding',    # dense visual embedding
    'instance',    # instance segmentation using bottom-up approach
    'orientation',     # orientation estimation for (selected) instances
    'normal',    # surface normal estimation
    'scene'    # scene classification
)

KNOWN_TOKEN_TASKS = (
    'token-mask',  # binary mask prediction using token-based decoder
    'token-semantic',  # semantic segmentation using token-based decoder
    'token-panoptic',  # panoptic segmentation using token-based decoder
    'token-orientation',  # orientation estimation using token-based decoder
    'token-scene',  # scene classification using token-based decoder
    'token-visual-embedding',  # visual embedding using token-based decoder
    'token-image-embedding'  # image embedding using token-based decoder
)
