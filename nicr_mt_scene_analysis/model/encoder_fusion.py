# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Optional, Tuple, Type

import torch.nn as nn

from ..utils import partial_class
from .activation import get_activation_class
from ..types import EncoderForwardType
from .utils import SqueezeAndExcitation


KNOWN_ENCODER_FUSIONS = (
    'se-add', 'add',    # bidirectional (note same features after first fusion)
    'add-uni-rgb', 'add-uni-depth',    # only unidirectional
    'se-add-uni-rgb', 'se-add-uni-depth',    # only unidirectional but with SE
    'none'    # do not fuse features
)


class EncoderFusionWeightedAdd(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        destinations: Tuple[str, ...],
        use_se_weighting: bool,
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs
    ) -> None:
        super().__init__()
        if use_se_weighting:
            # weight features using SqueezeAndExcitation before
            self.weighting_rgb = SqueezeAndExcitation(n_channels_in,
                                                      activation=activation)
            self.weighting_depth = SqueezeAndExcitation(n_channels_in,
                                                        activation=activation)
        else:
            # do not weight features
            self.weighting_rgb = nn.Identity()
            self.weighting_depth = nn.Identity()

        self._destinations = destinations

    def forward(self, x: EncoderForwardType) -> EncoderForwardType:
        x_rgb, x_depth = x

        # apply optional weighting
        rgb_weighted = self.weighting_rgb(x_rgb)
        depth_weighted = self.weighting_depth(x_depth)

        # fuse features
        fused = rgb_weighted + depth_weighted

        # determine where to return fused features
        y_rgb = fused if 'rgb' in self._destinations else x_rgb
        y_depth = fused if 'depth' in self._destinations else x_depth

        return y_rgb, y_depth


EncoderFusionType = EncoderFusionWeightedAdd


def get_encoder_fusion_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[EncoderFusionType]:
    # global default
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

    return partial_class(EncoderFusionWeightedAdd, **kwargs)
