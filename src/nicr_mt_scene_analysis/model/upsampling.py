# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Optional, Tuple, Type, Union
import warnings

import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import interpolate
from torch.nn.functional import layer_norm

from ..utils import partial_class


class LayerNorm2d(nn.LayerNorm):
    # per-channel LayerNorm on (B, C, H, W) tensors. nn.LayerNorm only
    # normalizes the last dim, so we permute to channels-last before the
    # functional call and back afterwards.
    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape, self.weight, self.bias, self.eps,
        ).permute(0, 3, 1, 2).contiguous()


KNOWN_UPSAMPLING_METHODS = (
    'nearest',    # nearest interpolation
    'bilinear',    # bilinear interpolation
    'learned-3x3',    # nearest + reflection padding + depth-wise conv
    'learned-3x3-zeropad',    # nearest + zero padding + depth-wise conv
    'transposed-conv',    # transposed conv + GELU + depthwise conv + LayerNorm2d
)


class TransposedConvUpsampling(nn.Module):
    # learned 2x upsampling block as described in EoMT (Kerssies et al., CVPR
    # 2025), which follows ViTDet's transposed-convolution upscaling (Li et
    # al., ECCV 2022). A 2x2 transposed convolution with stride 2, GELU, a
    # depthwise 3x3 convolution, and a final per-channel norm.
    def __init__(
        self,
        n_channels: int,
        mode: str = 'transposed-conv',  # to match signature of Upsampling
        scale_factor: Union[float, Tuple[float, float]] = 2.,
        use_bias: bool = True
    ):
        super().__init__()

        if mode != 'transposed-conv':
            warnings.warn(
                f"TransposedConvUpsampling ignores mode='{mode}' and always "
                "performs learned upsampling."
            )
        if scale_factor not in (2., (2., 2.)):
            warnings.warn(
                "TransposedConvUpsampling currently upsamples by a factor of "
                "2. The provided scale_factor argument is ignored."
            )

        self.conv1 = nn.ConvTranspose2d(
            n_channels,
            n_channels,
            kernel_size=2,
            stride=2,
            bias=use_bias
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            padding=1,
            groups=n_channels,
            bias=False,
        )
        self.norm = LayerNorm2d(n_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x


class Upsampling(nn.Module):
    def __init__(
        self,
        mode: str,
        n_channels: int,
        scale_factor: Union[float, Tuple[float, float]] = 2.,
        use_bias: bool = True
    ) -> None:
        super().__init__()

        if mode == 'bilinear':
            self._align_corners = False
        else:
            self._align_corners = None

        if 'learned-3x3' in mode:
            # mimics bilinear interpolation by first applying nearest neighbor
            # upscaling and subsequently a 3x3 depthwise conv with weights
            # realizing bilinear interpolation
            # note, only works as supposed when feature maps are upsampled by
            # a factor of 2
            assert scale_factor == 2. or scale_factor == (2., 2.)

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(n_channels, n_channels,
                                      groups=n_channels,
                                      kernel_size=3,
                                      padding=0,
                                      bias=use_bias)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(n_channels, n_channels,
                                      groups=n_channels,
                                      kernel_size=3,
                                      padding=1,
                                      bias=use_bias)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w]*n_channels))

            # set bias to zero
            if use_bias:
                with torch.no_grad():
                    self.conv.bias.zero_()

            self._mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self._mode = mode

        self._scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        # note that recently, onnx op requires a scale parameter
        # _, _, h, w = x.shape
        x = interpolate(x,
                        # size=(int(h*2), int(w*2)),
                        scale_factor=self._scale_factor,
                        mode=self._mode,
                        align_corners=self._align_corners)
        x = self.pad(x)
        x = self.conv(x)

        return x


UpsamplingType = Upsampling


def get_upsampling_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[UpsamplingType]:
    # global default
    if name is None:
        name = 'bilinear'

    name = name.lower()
    if name not in KNOWN_UPSAMPLING_METHODS:
        raise ValueError(f"Unknown upsampling: '{name}'")
    if name == 'transposed-conv':
        return partial_class(TransposedConvUpsampling, **kwargs)

    kwargs['mode'] = name
    return partial_class(Upsampling, **kwargs)
