# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from collections import OrderedDict
from math import log2
from typing import Optional, Tuple, Type

from torch import nn

from ...utils import NormalOutputNormalization
from ..activation import get_activation_class
from ..block import BlockType
from ..normalization import get_normalization_class
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from .dense_base import DenseDecoderBase


class NormalDecoder(DenseDecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, int, int],
        block: Type[BlockType],
        n_blocks: int,
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, int, int],
        n_channels_out: int = 3,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('normal'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(n_channels_in=n_channels_in,
                         downsampling_in=downsampling_in,
                         n_channels=n_channels,
                         block=block,
                         n_blocks=n_blocks,
                         fusion=fusion,
                         fusion_n_channels=fusion_n_channels,
                         postprocessing=postprocessing,
                         normalization=normalization,
                         activation=activation,
                         upsampling=upsampling)

        self._n_channels_out = n_channels_out

        # final task head
        self._task_head = self._create_task_head(
            n_channels_in=n_channels[-1],
            upsampling=prediction_upsampling,
            n_upsamplings=int(log2(downsampling_in / 2**len(n_channels)))
        )

        # heads for side outputs
        side_output_heads = []
        for n in n_channels:
            side_output_heads.append(self._create_task_head(n))
        self._side_output_heads = nn.ModuleList(side_output_heads)

    def _create_task_head(
        self,
        n_channels_in: int,
        upsampling: Optional[Type[UpsamplingType]] = None,
        n_upsamplings: int = 0
    ) -> nn.Module:
        modules = []
        # convolution mapping to xyz
        # note, we use a 1x1 conv instead of a 3x3 conv for side outputs
        is_main_output = n_upsamplings != 0
        modules.append((
            'conv',
            nn.Conv2d(n_channels_in, self._n_channels_out,
                      kernel_size=3 if is_main_output else 1,
                      padding=1 if is_main_output else 0)
        ))

        # prediction upsampling
        for i in range(n_upsamplings):
            modules.append((
                f'upsample_{i}',
                upsampling(n_channels=self._n_channels_out)
            ))

        # normalize to have unit length
        modules.append(('act', NormalOutputNormalization()))

        return nn.Sequential(OrderedDict(modules))

    @property
    def task_head(self) -> nn.Module:
        return self._task_head

    @property
    def side_output_heads(self) -> nn.ModuleList:
        return self._side_output_heads
