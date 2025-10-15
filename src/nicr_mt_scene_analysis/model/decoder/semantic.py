# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Type

from math import log2

from torch import nn

from ..activation import get_activation_class
from ..block import BlockType
from .dense_utils import create_task_head
from ..normalization import get_normalization_class
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from .dense_base import DenseDecoderBase
from .mlp_base import MLPDecoderBase


class SemanticDecoder(DenseDecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        downsamplings: Tuple[int, ...],
        block: Type[BlockType],
        n_blocks: int,
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        n_classes: int,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('semantic'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(n_channels_in=n_channels_in,
                         downsampling_in=downsampling_in,
                         n_channels=n_channels,
                         downsamplings=downsamplings,
                         block=block,
                         n_blocks=n_blocks,
                         fusion=fusion,
                         fusion_n_channels=fusion_n_channels,
                         fusion_downsamplings=fusion_downsamplings,
                         postprocessing=postprocessing,
                         normalization=normalization,
                         activation=activation,
                         upsampling=upsampling)

        self._n_classes = n_classes

        # final task head
        self._task_head = create_task_head(
            n_channels_in=n_channels[-1],
            n_channels_out=self._n_classes,
            upsampling=prediction_upsampling,
            n_upsamplings=int(log2(downsamplings[-1]))
        )

        # heads for side outputs
        side_output_heads = []
        for n in self.side_output_n_channels:
            side_output_heads.append(
                create_task_head(n_channels_in=n, n_channels_out=self._n_classes)
            )
        self._side_output_heads = nn.ModuleList(side_output_heads)

    @property
    def task_head(self) -> nn.Module:
        return self._task_head

    @property
    def side_output_heads(self) -> nn.ModuleList:
        return self._side_output_heads


class SemanticMLPDecoder(MLPDecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        n_classes: int,
        downsampling_in_heads: int = 4,
        dropout_p: float = 0.1,
        n_channels_out: Optional[int] = None,
        n_upsamplings: Optional[int] = None,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('semantic'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(n_channels_in=n_channels_in,
                         downsampling_in=downsampling_in,
                         n_channels=n_channels,
                         fusion=fusion,
                         fusion_n_channels=fusion_n_channels,
                         fusion_downsamplings=fusion_downsamplings,
                         downsampling_in_heads=downsampling_in_heads,
                         dropout_p=dropout_p,
                         postprocessing=postprocessing,
                         normalization=normalization,
                         activation=activation,
                         upsampling=upsampling)

        self._n_classes = n_classes
        self._downsampling_in_heads = downsampling_in_heads

        # Calculate default n_channels_out if not provided
        if n_channels_out is None:
            n_channels_out = sum(n_channels)//len(n_channels)

        # Calculate default n_upsamplings if not provided
        if n_upsamplings is None:
            n_upsamplings = self._downsampling_in_heads//2

        # final task head
        self._task_head = create_task_head(
            n_channels_in=n_channels_out,
            n_channels_out=self._n_classes,
            upsampling=prediction_upsampling,
            n_upsamplings=n_upsamplings,   # each doubles res.
        )

    @property
    def task_head(self) -> nn.Module:
        return self._task_head
