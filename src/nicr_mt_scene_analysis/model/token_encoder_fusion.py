# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Optional, Tuple, Type

import torch.nn as nn

from ..types import EncoderForwardType
from ..utils import partial_class
from .activation import get_activation_class
from .encoder_fusion import KNOWN_ENCODER_FUSIONS
from .utils import TokenSqueezeAndExcitation


class TokenEncoderRGBDFusionWeightedAdd(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        destinations: Tuple[str, ...],
        use_se_weighting: bool,
        input_memory_layout: str = 'BND',
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs
    ):
        super().__init__()

        assert input_memory_layout == 'BND', (
            f"TokenEncoderRGBDFusionWeightedAdd expects 'BND', got "
            f"'{input_memory_layout}'."
        )

        if use_se_weighting:
            self.weighting_rgb = TokenSqueezeAndExcitation(
                n_channels_in, activation=activation
            )
            self.weighting_depth = TokenSqueezeAndExcitation(
                n_channels_in, activation=activation
            )

        self._n_channels_in = n_channels_in
        self._use_se_weighting = use_se_weighting
        self._destinations = destinations
        self._input_memory_layout = input_memory_layout

    def forward(self, x: EncoderForwardType) -> EncoderForwardType:
        x_rgb, x_depth = x['rgb'], x['depth']

        if self._use_se_weighting:
            rgb_weighted = self.weighting_rgb(x_rgb)
            depth_weighted = self.weighting_depth(x_depth)
        else:
            rgb_weighted = x_rgb
            depth_weighted = x_depth

        fused = rgb_weighted + depth_weighted

        y_rgb = fused if 'rgb' in self._destinations else x_rgb
        y_depth = fused if 'depth' in self._destinations else x_depth

        return {'rgb': y_rgb, 'depth': y_depth}


TokenEncoderFusionType = TokenEncoderRGBDFusionWeightedAdd


def get_token_encoder_fusion_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[TokenEncoderFusionType]:
    if name is None:
        name = 'add-uni-rgb'

    name = name.lower()
    if name not in KNOWN_ENCODER_FUSIONS:
        raise ValueError(f"Unknown encoder fusion: '{name}'")

    kwargs['use_se_weighting'] = 'se' in name

    if 'uni-rgb' in name:
        kwargs['destinations'] = ('rgb',)
    elif 'uni-depth' in name:
        kwargs['destinations'] = ('depth',)
    elif 'none' == name:
        kwargs['destinations'] = ()
    else:
        kwargs['destinations'] = ('rgb', 'depth',)

    return partial_class(TokenEncoderRGBDFusionWeightedAdd, **kwargs)
