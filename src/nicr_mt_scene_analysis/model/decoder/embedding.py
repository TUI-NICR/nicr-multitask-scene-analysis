# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Type

from math import log2

from torch import nn

from ..activation import get_activation_class
from ..block import BlockType
from ..normalization import get_normalization_class
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from .dense_utils import create_task_head
from .dense_base import DenseDecoderBase
from .mlp_base import MLPDecoderBase


class EmbeddingDecoder(DenseDecoderBase):
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
        embedding_dim: int,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('dense-visual-embedding'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class(),
        n_upsamplings: Optional[int] = None
    ) -> None:
        super().__init__(
            n_channels_in=n_channels_in,
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
            upsampling=upsampling
        )

        self._embedding_dim = embedding_dim

        # calculate default n_upsamplings if not provided
        if n_upsamplings is None:
            n_upsamplings = int(log2(downsamplings[-1]))

        # final task head
        self._task_head = create_task_head(
            n_channels_in=n_channels[-1],
            n_channels_out=self._embedding_dim,
            upsampling=prediction_upsampling,
            n_upsamplings=n_upsamplings
        )

        # heads for side outputs
        side_output_heads = []
        for n in self.side_output_n_channels:
            side_output_heads.append(
                create_task_head(n_channels_in=n, n_channels_out=self._embedding_dim)
            )
        self._side_output_heads = nn.ModuleList(side_output_heads)

    @property
    def task_head(self) -> nn.Module:
        return self._task_head

    @property
    def side_output_heads(self) -> nn.ModuleList:
        return self._side_output_heads


class EmbeddingMLPDecoder(MLPDecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        embedding_dim: int,
        downsampling_in_heads: int = 4,
        dropout_p: float = 0.1,
        n_channels_out: Optional[int] = None,
        n_upsamplings: Optional[int] = None,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('dense-visual-embedding'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        # Calculate default n_channels_out if not provided
        if n_channels_out is None:
            n_channels_out = sum(n_channels)//len(n_channels)

        super().__init__(
            n_channels_in=n_channels_in,
            downsampling_in=downsampling_in,
            n_channels=n_channels,
            fusion=fusion,
            fusion_n_channels=fusion_n_channels,
            fusion_downsamplings=fusion_downsamplings,
            downsampling_in_heads=downsampling_in_heads,
            dropout_p=dropout_p,
            n_channels_out=n_channels_out,
            postprocessing=postprocessing,
            normalization=normalization,
            activation=activation,
            upsampling=upsampling
        )

        self._embedding_dim = embedding_dim
        self._downsampling_in_heads = downsampling_in_heads

        # calculate default n_upsamplings if not provided
        if n_upsamplings is None:
            n_upsamplings = self._downsampling_in_heads//2

        # final task head
        self._task_head = create_task_head(
            n_channels_in=n_channels_out,
            n_channels_out=self._embedding_dim,
            upsampling=prediction_upsampling,
            n_upsamplings=n_upsamplings
        )

    @property
    def task_head(self) -> nn.Module:
        return self._task_head
