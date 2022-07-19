# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.backbone.pretraining import ImageNetClassifier
from nicr_mt_scene_analysis.testing.model import save_ckpt
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


def create_classifier_ckpt(filepath, name, block_name, n_input_channels,
                           **kwargs):
    backbone = get_backbone(name, block_name,
                            n_input_channels=n_input_channels,
                            pretrained=False,
                            **kwargs)
    classifier = ImageNetClassifier(backbone)
    save_ckpt(filepath, model=classifier)


@pytest.mark.parametrize('name', ('resnet18', 'resnet34', 'resnet50',
                                  'resnet101'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (False, True))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
def test_resnet(name, n_input_channels, pretrained, activation, tmp_path):
    """Test original ResNet"""

    if 'resnet18' in name or 'resnet34' in name:
        block_name = 'basicblock'
    else:
        block_name = 'bottleneck'

    model = get_backbone(name, block_name,
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         activation=activation)

    x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += f'__block_{block_name}'
    filename += f'__act_{activation}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('name', ('resnet18', 'resnet34'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (True, False))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
def test_resnet_nonbottleneck1d(name, n_input_channels, pretrained, activation,
                                tmp_path):
    """Test ResNet18 / ResNet34 with NonBottleneck1D"""
    if pretrained:
        # there are no pretrained weights, create them
        pretrained_filepath = os.path.join(tmp_path,
                                           f'{name}_nonbottleneck1d.pth')
        create_classifier_ckpt(pretrained_filepath, name, 'nonbottleneck1d',
                               n_input_channels,
                               activation=activation)
        kwargs = {
            'pretrained_filepath': pretrained_filepath
        }
    else:
        kwargs = {}

    model = get_backbone(name, 'nonbottleneck1d',
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         activation=activation,
                         **kwargs)

    x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += f'__block_nonbottleneck1d'
    filename += f'__act_{activation}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('name', ('resnet18se', 'resnet34se', 'resnet50se',
                                  'resnet101se'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (False, True))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
def test_resnetse(name, n_input_channels, pretrained, activation, tmp_path):
    """Test ResNetSE"""
    if 'resnet18' in name or 'resnet34' in name:
        block_name = 'basicblock'
    else:
        block_name = 'bottleneck'

    if pretrained:
        # there are no pretrained weights, create them
        pretrained_filepath = os.path.join(tmp_path,
                                           f'{name}_{block_name}.pth')
        create_classifier_ckpt(pretrained_filepath, name, block_name,
                               n_input_channels,
                               activation=activation)
        kwargs = {
            'pretrained_filepath': pretrained_filepath
        }
    else:
        kwargs = {}

    model = get_backbone(name, block_name,
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         activation=activation,
                         **kwargs)

    x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += f'__block_{block_name}'
    filename += f'__act_{activation}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('name', ('resnet18se', 'resnet34se'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (True, False))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
def test_resnetse_nonbottleneck1d(name, n_input_channels, pretrained,
                                  activation, tmp_path):
    """Test ResNet18(SE) / ResNet34(SE) with NonBottleneck1D"""
    if pretrained:
        # there are no pretrained weights, create them
        pretrained_filepath = os.path.join(tmp_path,
                                           f'{name}_nonbottleneck1d.pth')
        create_classifier_ckpt(pretrained_filepath, name, 'nonbottleneck1d',
                               n_input_channels,
                               activation=activation)
        kwargs = {
            'pretrained_filepath': pretrained_filepath
        }
    else:
        kwargs = {}

    model = get_backbone(name, 'nonbottleneck1d',
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         activation=activation,
                         **kwargs)

    x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += f'__block_nonbottleneck1d'
    filename += f'__act_{activation}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)
