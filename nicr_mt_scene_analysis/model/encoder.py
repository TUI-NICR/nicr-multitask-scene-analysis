# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from itertools import compress
from typing import Sequence, Tuple, Type, Union

from torch import nn

from .activation import get_activation_class
from .backbone.base import Backbone
from .encoder_fusion import EncoderFusionType
from .encoder_fusion import get_encoder_fusion_class
from .normalization import get_normalization_class
from ..types import EncoderInputType
from ..types import EncoderOutputType


class FusedEncoders(nn.Module):
    def __init__(
        self,
        backbone_rgb: Union[Backbone, None],
        backbone_depth: Union[Backbone, None],
        fusion: Type[EncoderFusionType],
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        skip_downsamplings: Sequence[int] = (4, 8, 16)
    ) -> None:
        super().__init__()

        # store backbones
        self.backbone_rgb = backbone_rgb
        self.backbone_depth = backbone_depth

        # create fusion modules for RGBD
        if self.backbone_rgb is not None and self.backbone_depth is not None:
            # check some assumptions
            b_rgb = self.backbone_rgb
            b_depth = self.backbone_depth
            assert len(b_rgb.stages) == len(b_depth.stages)
            assert b_rgb.stages_n_channels == b_depth.stages_n_channels
            assert b_rgb.stages_downsampling == b_depth.stages_downsampling

            fusions = []
            for n in b_rgb.stages_n_channels:
                fusions.append(fusion(n_channels_in=n,
                                      normalization=normalization,
                                      activation=activation))
            self.fusions = nn.ModuleList(fusions)

        # compute some stuff for forward propagation
        bb = self.backbone_rgb or self.backbone_depth
        self._n_stages = len(bb.stages)
        self._stages_downsampling = bb.stages_downsampling
        self._stages_n_channels = bb.stages_n_channels

        # determine where to create skip connections
        self._skips_downsamplings = skip_downsamplings
        self._stages_skip_connections = [False] * self._n_stages
        for ds in self._skips_downsamplings:
            last_idx = self._stages_downsampling[::-1].index(ds)
            stage_idx = self._n_stages - 1 - last_idx
            self._stages_skip_connections[stage_idx] = True

    @property
    def skips_n_channels(self) -> Tuple[int]:
        return tuple(compress(data=self._stages_n_channels,
                              selectors=self._stages_skip_connections))

    @property
    def skips_downsamplings(self) -> Sequence[int]:
        return self._skips_downsamplings

    @property
    def n_channels_out(self) -> int:
        return self._stages_n_channels[-1]

    @property
    def downsampling(self):
        return self._stages_downsampling[-1]

    def forward(self, x: EncoderInputType) -> EncoderOutputType:
        x_rgb, x_depth = x

        # feature for skip connections
        decoder_skips = []

        # forward stages
        backbone_rgb = self.backbone_rgb
        backbone_depth = self.backbone_depth
        for idx in range(self._n_stages):
            # forward rgb stage
            if backbone_rgb is not None:
                x_rgb = backbone_rgb.forward_stage(idx, x_rgb)

            # forward depth stage
            if backbone_depth is not None:
                x_depth = backbone_depth.forward_stage(idx, x_depth)

            # fuse both outputs
            if backbone_rgb is not None and backbone_depth is not None:
                x_rgb, x_depth = self.fusions[idx]((x_rgb, x_depth))

            # check if features for a skip connection need to be stored
            if self._stages_skip_connections[idx]:
                decoder_skips.append((x_rgb, x_depth))

        return (x_rgb, x_depth), tuple(decoder_skips)


EncoderType = FusedEncoders


def get_fused_encoders(
    backbone_rgb: Union[Backbone, None],
    backbone_depth: Union[Backbone, None],
    fusion: str = 'add-uni-rgb',
    normalization: str = 'batchnorm',
    activation: str = 'relu',
    skip_downsamplings: Sequence[int] = (4, 8, 16)
) -> EncoderType:

    # create encoders
    encoders = FusedEncoders(
        backbone_rgb=backbone_rgb,
        backbone_depth=backbone_depth,
        fusion=get_encoder_fusion_class(fusion),
        normalization=get_normalization_class(normalization),
        activation=get_activation_class(activation),
        skip_downsamplings=skip_downsamplings
    )

    return encoders
