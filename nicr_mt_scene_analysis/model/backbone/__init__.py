# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from collections import OrderedDict
import os
from typing import Any, Optional, Type, Union

import torch

from ..activation import get_activation_class
from ..block import BlockType
from ..block import BasicBlock
from ..block import Bottleneck
from ..block import get_block_class
from ..normalization import get_normalization_class
from .base import Backbone
from .resnet import get_resnet_backbone


KNOWN_BACKBONES = [
    # ResNet (v1)
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    # ResNet (v1) +SE at the end of each stage
    'resnet18se', 'resnet34se', 'resnet50se', 'resnet101se',
]


def get_backbone(
    name: str,
    block: Union[str, Type[BlockType]],
    n_input_channels: int = 3,
    normalization: Union[str, Type[torch.nn.Module]] = 'batchnorm',
    activation: Union[str, Type[torch.nn.Module]] = 'relu',
    pretrained: bool = True,
    pretrained_filepath: Optional[str] = None,
    **kwargs: Any
) -> Backbone:
    name = name.lower()

    # get block class from str
    if isinstance(block, str):
        block = get_block_class(block)

    # get normalization and activation classes
    if isinstance(normalization, str):
        normalization = get_normalization_class(normalization)
    if isinstance(activation, str):
        activation = get_activation_class(activation)

    pretrained_torchvision = pretrained

    if 'resnet' in name:
        pretrained_torchvision &= not name.endswith('se')
        pretrained_torchvision &= issubclass(block, (BasicBlock, Bottleneck))

        backbone = get_resnet_backbone(
            name, block,
            pretrained_torchvision=pretrained_torchvision,
            normalization=normalization,
            activation=activation,
            n_input_channels=n_input_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if pretrained and not pretrained_torchvision:
        print(f"Loading pretrained weights from: '{pretrained_filepath}'")
        checkpoint = torch.load(pretrained_filepath,
                                map_location=lambda storage, loc: storage)

        state_dict = OrderedDict()
        # remove "backbone." from weights
        for key, weight in checkpoint['state_dict'].items():
            new_key = key.replace('model.', '')
            new_key = new_key.replace('backbone.', '')
            state_dict[new_key] = weight

        # remove keys of final fully-connected layer
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        # sum the weights of the first convolution for depth
        if n_input_channels == 1:
            state_dict['conv1.weight'] = torch.sum(
                state_dict['conv1.weight'],
                axis=1,
                keepdim=True
            )

        backbone.load_state_dict(state_dict, strict=True)

    return backbone
