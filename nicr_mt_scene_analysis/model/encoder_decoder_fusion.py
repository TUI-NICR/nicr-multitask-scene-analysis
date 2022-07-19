# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

TODO: write some encoder decoder fusion tests
"""
from typing import Any, Optional, Type, Union

from torch import Tensor
import torch.nn as nn

from ..utils import partial_class
from .activation import get_activation_class
from ..types import EncoderSkipType
from .normalization import get_normalization_class
from .utils import ConvNormAct


KNOWN_ENCODER_DECODER_FUSIONS = (
    'add-rgb', 'add-depth',    # add rgb/depth encoder features
    'none'
)


class EncoderDecoderFusionAdd(nn.Module):
    def __init__(
        self,
        n_channels_encoder: int,
        n_channels_decoder: int,
        fuse_features_from: Union[str, None],
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class()
    ) -> None:
        super().__init__()

        if n_channels_encoder != n_channels_decoder:
            self.layer = ConvNormAct(n_channels_encoder, n_channels_decoder,
                                     normalization=normalization,
                                     activation=activation)
        else:
            self.layer = nn.Identity()

        self._fuse_features_from = fuse_features_from

    def forward(
        self,
        x_enc: EncoderSkipType,
        x_dec: Tensor
    ) -> Tensor:
        x_enc_rgb, x_enc_depth = x_enc

        if 'rgb' == self._fuse_features_from:
            out = x_dec + self.layer(x_enc_rgb)
        elif 'depth' == self._fuse_features_from:
            out = x_dec + self.layer(x_enc_depth)
        elif self._fuse_features_from is None:
            out = x_dec
        else:
            raise ValueError()

        return out


EncoderDecoderFusionType = EncoderDecoderFusionAdd


def get_encoder_decoder_fusion_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[EncoderDecoderFusionType]:
    # global default
    if name is None:
        name = 'add-rgb'

    name = name.lower()
    if name not in KNOWN_ENCODER_DECODER_FUSIONS:
        raise ValueError(f"Unknown encoder decoder fusion: '{name}'")

    if 'add-rgb' == name:
        kwargs['fuse_features_from'] = 'rgb'
    elif 'add-depth' == name:
        kwargs['fuse_features_from'] = 'depth'
    elif 'none' == name:
        kwargs['fuse_features_from'] = None

    return partial_class(EncoderDecoderFusionAdd, **kwargs)
