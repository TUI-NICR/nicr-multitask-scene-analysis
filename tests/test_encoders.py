# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from itertools import product
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.encoder import get_fused_encoders
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


def fused_encoder_test(modalities,
                       backbone_rgb,
                       backbone_depth,
                       activation,
                       fusion,
                       tmp_path):
    # build encoder model
    if 'resnet50' in backbone_rgb:
        backbone_rgb_block = 'bottleneck'
    else:
        backbone_rgb_block = 'nonbottleneck1d'

    if 'resnet50' in backbone_depth:
        backbone_depth_block = 'bottleneck'
    else:
        backbone_depth_block = 'nonbottleneck1d'

    decoder_skip_downsamplings = (4, 8, 16)

    if 'rgb' in modalities:
        x_rgb = torch.randn((2, 3, 480, 640))
        backbone_rgb = get_backbone(
            name=backbone_rgb,
            block=backbone_rgb_block,
            n_input_channels=3,
            normalization='batchnorm',
            activation=activation,
            pretrained=False,
        )
    else:
        x_rgb = None
        backbone_rgb = None

    if 'depth' in modalities:
        x_depth = torch.randn((2, 1, 480, 640))
        backbone_depth = get_backbone(
            name=backbone_depth,
            block=backbone_depth_block,
            n_input_channels=1,
            normalization='batchnorm',
            activation=activation,
            pretrained=False,
        )
    else:
        x_depth = None
        backbone_depth = None

    encoder = get_fused_encoders(
        backbone_rgb=backbone_rgb,
        backbone_depth=backbone_depth,
        fusion=fusion,
        normalization='batchnorm',
        activation=activation,
        skip_downsamplings=decoder_skip_downsamplings
    )

    # apply model
    encoder((x_rgb, x_depth))

    # export model
    modalities_str = '+'.join(modalities)
    filename = f'encoder_{modalities_str}'
    filename += f'__backbone_rgb_{backbone_rgb}'
    filename += f'__backbone_depth_{backbone_depth}'
    filename += f'__act_{activation}'
    filename += f'__fusion_{fusion}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = [f'input_{m}' for m in modalities]
    output_names = [f'output_{m}' for m in modalities]
    output_names += [f'skip_{ds}_{m}'
                     for ds, m in product(decoder_skip_downsamplings,
                                          modalities)]
    export_onnx_model(filepath, encoder, ((x_rgb, x_depth),),
                      input_names=input_names,
                      output_names=output_names)


@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', ('resnet18', 'resnet50', 'resnet34se'))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoders1(modalities, backbone, activation, fusion,
                         tmp_path):
    """Test FusedEncoders with same backbone"""
    fused_encoder_test(modalities=modalities,
                       backbone_rgb=backbone,
                       backbone_depth=backbone,
                       activation=activation,
                       fusion=fusion,
                       tmp_path=tmp_path)


@pytest.mark.parametrize('backbones', (('resnet18', 'resnet34'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoders2(backbones, activation, fusion,
                         tmp_path):
    """Test FusedEncoders with different backbone"""
    fused_encoder_test(modalities=('rgb', 'depth'),
                       backbone_rgb=backbones[0],
                       backbone_depth=backbones[1],
                       activation=activation,
                       fusion=fusion,
                       tmp_path=tmp_path)


@pytest.mark.xfail(reason='Not yet implemented')
@pytest.mark.parametrize('backbones', (('resnet18', 'resnet50'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoders3(backbones, activation, fusion,
                         tmp_path):
    """Test FusedEncoders with different backbone"""
    fused_encoder_test(modalities=('rgb', 'depth'),
                       backbone_rgb=backbones[0],
                       backbone_depth=backbones[1],
                       activation=activation,
                       fusion=fusion,
                       tmp_path=tmp_path)
