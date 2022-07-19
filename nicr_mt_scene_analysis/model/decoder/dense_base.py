# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import Tuple, Type

import torch.nn as nn
from torch import Tensor

from ..activation import get_activation_class
from ..block import BlockType
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..normalization import get_normalization_class
from ..postprocessing import PostProcessingType
from ...types import DecoderInputType
from ...types import DenseDecoderModuleOutputType
from ...types import DecoderRawOutputType
from ...types import EncoderSkipsType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..utils import ConvNormAct
from .base import DecoderBase


class DenseDecoderModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels: int,
        block: Type[BlockType],
        n_blocks: int,
        initial_conv: bool = True,
        activation: Type[nn.Module] = get_activation_class(),
        normalization: Type[nn.Module] = get_normalization_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__()

        # initial conv
        if initial_conv:
            self.conv = ConvNormAct(n_channels_in, n_channels,
                                    kernel_size=3,
                                    normalization=normalization,
                                    activation=activation)
            blocks_n_channels = [n_channels] * (n_blocks+1)
        else:
            self.conv = nn.Identity()
            blocks_n_channels = [n_channels_in] + [n_channels] * n_blocks
            assert n_blocks > 0

        # subsequent blocks
        blocks = []
        for i in range(n_blocks):
            if blocks_n_channels[i] != blocks_n_channels[i+1]:
                # skip connections need to modify the number of channels
                downsample = ConvNormAct(
                    blocks_n_channels[i], blocks_n_channels[i+1],
                    kernel_size=1,
                    activation=None
                )
            else:
                downsample = None

            blocks.append(
                block(inplanes=blocks_n_channels[i],
                      planes=blocks_n_channels[i+1],
                      stride=1,
                      downsample=downsample,
                      groups=1,
                      base_width=64,
                      dilation=1,
                      normalization=normalization,
                      activation=activation)
            )
        self.blocks = nn.Sequential(*blocks)

        # final upsampling
        self.upsample = upsampling(n_channels=n_channels)

    def forward(self, x: Tensor) -> DenseDecoderModuleOutputType:
        out = self.conv(x)
        out = self.blocks(out)

        # for pyramid supervision
        out_side = out if self.training else None

        out = self.upsample(out)

        return out, out_side


class DenseDecoderBase(DecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, int, int],
        block: Type[BlockType],
        n_blocks: int,
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, int, int],
        postprocessing: Type[PostProcessingType],
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(postprocessing=postprocessing)

        # decoder modules
        decoder_modules = []
        for n_in, n_out in zip((n_channels_in,) + n_channels[:-1],
                               n_channels):
            decoder_modules.append(
                DenseDecoderModule(
                    n_channels_in=n_in,
                    n_channels=n_out,
                    block=block,
                    n_blocks=n_blocks,
                    activation=activation,
                    normalization=normalization,
                    upsampling=upsampling
                )
            )
        self.decoder_modules = nn.ModuleList(decoder_modules)
        self._side_output_downscales = tuple(
            int(downsampling_in / (2 ** i))
            for i in range(len(decoder_modules))
        )

        # fusion modules
        fusions = []
        for n_skip, n_dec in zip(fusion_n_channels, n_channels):
            fusions.append(
                fusion(
                    n_channels_encoder=n_skip,
                    n_channels_decoder=n_dec,
                    activation=activation,
                    normalization=normalization
                )
            )
        self.fusions = nn.ModuleList(fusions)

        assert len(self.decoder_modules) == len(self.fusions)

    @property
    @abc.abstractmethod
    def task_head(self) -> nn.Module:
        pass

    @property
    @abc.abstractmethod
    def side_output_heads(self) -> nn.ModuleList:
        pass

    @property
    def side_output_downscales(self) -> Tuple[int]:
        return self._side_output_downscales

    def _forward_decoder_modules(
        self,
        x: Tensor,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # some checks
        assert len(skips) == len(self.fusions)

        side_outputs = []
        for dec_m, fusion, skip in zip(self.decoder_modules,
                                       self.fusions,
                                       skips):
            # apply decoder module
            x, side_outs = dec_m(x)
            side_outputs.append(side_outs)

            # apply encoder-decoder fusion
            x = fusion(x_enc=skip, x_dec=x)

        return x, tuple(side_outputs)

    def _forward_training(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # we are only interested in the main output, not in the context features
        x, _ = x

        # apply decoder modules
        output, side_outputs = self._forward_decoder_modules(x=x,
                                                             skips=skips)

        # apply task head and final upsampling
        output = self.task_head(output)

        # apply side output heads
        side_outputs = [
            side_head(side_out) if side_out is not None else None
            for side_head, side_out in zip(self.side_output_heads, side_outputs)
        ]

        return output, tuple(side_outputs)
