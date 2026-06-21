# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from itertools import product
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_mt_scene_analysis.model.backbone import IS_SWIN_AVAILABLE
from nicr_mt_scene_analysis.model.backbone import IS_TIMM_AVAILABLE
from nicr_mt_scene_analysis.model.backbone import KNOWN_TOKEN_BACKBONES


def _build_backbone(name, *, n_input_channels, pretrained, resnet_block,
                    normalization, activation):
    if name in KNOWN_TOKEN_BACKBONES:
        return get_token_backbone(
            name=name,
            n_input_channels=n_input_channels,
            pretrained=pretrained,
        )
    return get_backbone(
        name=name,
        resnet_block=resnet_block,
        n_input_channels=n_input_channels,
        normalization=normalization,
        activation=activation,
        pretrained=pretrained,
    )
from nicr_mt_scene_analysis.model.backbone.base import TokenBackbone
from nicr_mt_scene_analysis.model.encoder import ENCODER_META_KEY
from nicr_mt_scene_analysis.model.encoder import get_encoder
from nicr_mt_scene_analysis.model.token_encoder import get_token_encoder
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
from nicr_mt_scene_analysis.testing import EXPORT_ONNX_MODELS


def encoder_test1(modalities,
                  backbone_rgb,
                  backbone_depth,
                  activation,
                  fusion,
                  normalization,
                  tmp_path,
                  decoder_skip_downsamplings = (4, 8, 16)):
    # build encoder model
    if 'resnet50' in backbone_rgb:
        backbone_rgb_block = 'bottleneck'
    else:
        backbone_rgb_block = 'nonbottleneck1d'

    if 'resnet50' in backbone_depth:
        backbone_depth_block = 'bottleneck'
    else:
        backbone_depth_block = 'nonbottleneck1d'

    encoder_input = {}

    if 'rgb' in modalities:
        x_rgb = torch.randn((2, 3, 480, 640))
        encoder_input['rgb'] = x_rgb

        backbone_rgb = _build_backbone(
            name=backbone_rgb,
            n_input_channels=3,
            pretrained=False,
            resnet_block=backbone_rgb_block,
            normalization=normalization,
            activation=activation,
        )
        # required for dinov2 backbones to set input size for proper
        # pos-embedding
        if hasattr(backbone_rgb, 'set_input_size'):
            backbone_rgb.set_input_size(height=480, width=640)
    else:
        backbone_rgb = None

    if 'depth' in modalities:
        x_depth = torch.randn((2, 1, 480, 640))
        encoder_input['depth'] = x_depth

        backbone_depth = _build_backbone(
            name=backbone_depth,
            n_input_channels=1,
            pretrained=False,
            resnet_block=backbone_depth_block,
            normalization=normalization,
            activation=activation,
        )
        # required for dinov2 backbones to set input size for proper
        # pos-embedding
        if hasattr(backbone_depth, 'set_input_size'):
            backbone_depth.set_input_size(height=480, width=640)
    else:
        backbone_depth = None

    # create encoder. token backbones go through the token factory so SE
    # fusion picks up TokenSqueezeAndExcitation. extra-token indices are
    # required by the constructor but unused since no extra_tokens are passed.
    is_token = isinstance(backbone_rgb, TokenBackbone) or isinstance(
        backbone_depth, TokenBackbone
    )
    if is_token:
        if backbone_rgb is not None:
            ref_backbone = backbone_rgb
        else:
            ref_backbone = backbone_depth
        n_stages = len(ref_backbone.stages)
        encoder = get_token_encoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            backbone_rgbd=None,
            fusion=fusion,
            normalization=normalization,
            activation=activation,
            skip_downsamplings=decoder_skip_downsamplings,
            extra_tokens_start_stage_idx=0,
            extra_tokens_end_stage_idx=n_stages - 1
        )
    else:
        encoder = get_encoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            backbone_rgbd=None,
            fusion=fusion,
            normalization=normalization,
            activation=activation,
            skip_downsamplings=decoder_skip_downsamplings
        )

    # apply model
    encoder_output = encoder(encoder_input)

    # some simple checks
    keys = sorted(encoder_input.keys())
    # -> we have a tuple (output, skips), both elements are tuples
    assert len(encoder_output) == 2
    assert all(isinstance(eo, dict) for eo in encoder_output)
    # -> all modality keys in output must still be present (token encoders
    #    also inject an ENCODER_META_KEY entry which we ignore here)
    output_modality_keys = sorted(
        k for k in encoder_output[0] if k != ENCODER_META_KEY
    )
    assert len(output_modality_keys) == len(encoder_input)
    assert output_modality_keys == keys
    # -> skip connections must be a dict of dicts that as well contain all
    #    keys (token encoders tag each skip entry with `_stage_idx` which we
    #    ignore here, same as ENCODER_META_KEY above)
    assert len(encoder_output[1]) == len(decoder_skip_downsamplings)
    assert all(
        sorted(k for k in encoder_output[1][str(ds)] if not k.startswith('_'))
        == keys
        for ds in decoder_skip_downsamplings
    )

    # export model
    modalities_str = '+'.join(modalities)
    filename = f'encoder_{modalities_str}'
    # TODO: get type names instead of str representation.
    # str representation might contain the whole model structure
    # which is not good for filenames. Larger models might even exceed
    # filesystem limits.
    filename += f'__backbone_rgb_{type(backbone_rgb).__name__}'
    filename += f'__backbone_depth_{type(backbone_depth).__name__}'
    filename += f'__act_{activation}'
    filename += f'__fusion_{fusion}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = [f'input_{m}' for m in modalities]
    output_names = [f'output_{m}' for m in modalities]
    output_names += [f'skip_{ds}_{m}'
                     for ds, m in product(decoder_skip_downsamplings,
                                          modalities)]
    export_onnx_model(filepath, encoder, (encoder_input, {}),
                      input_names=input_names,
                      output_names=output_names)


def encoder_test2(modalities,
                  backbone_rgbd,
                  normalization,
                  tmp_path,
                  decoder_skip_downsamplings=(4, 8, 16)):
    # build encoder model

    encoder_input = {}

    x_rgbd = torch.randn((2, 4, 480, 640))
    encoder_input['rgbd'] = x_rgbd

    backbone_rgbd = _build_backbone(
        name=backbone_rgbd,
        n_input_channels=4,
        pretrained=False,
        resnet_block=None,
        normalization=normalization,
        activation='relu',
    )

    # required for dinov2 backbones to set input size for proper pos-embedding
    if hasattr(backbone_rgbd, 'set_input_size'):
        backbone_rgbd.set_input_size(height=480, width=640)

    # create encoder
    encoder = get_encoder(
        backbone_rgb=None,
        backbone_depth=None,
        backbone_rgbd=backbone_rgbd,
        normalization=normalization,
        activation='relu',
        skip_downsamplings=decoder_skip_downsamplings
    )

    # apply model
    encoder_output = encoder(encoder_input)

    # some simple checks
    keys = sorted(encoder_input.keys())
    # -> we have a tuple (output, skips), both elements are tuples
    assert len(encoder_output) == 2
    assert all(isinstance(eo, dict) for eo in encoder_output)
    # -> all keys in output must still be present
    assert len(encoder_output[0]) == len(encoder_input)
    assert sorted(encoder_output[0].keys()) == keys
    # -> skip connections must be a dict of dicts that as well contain all
    #    keys (token encoders tag each skip entry with `_stage_idx` which we
    #    ignore here, same as ENCODER_META_KEY above)
    assert len(encoder_output[1]) == len(decoder_skip_downsamplings)
    assert all(
        sorted(k for k in encoder_output[1][str(ds)] if not k.startswith('_'))
        == keys
        for ds in decoder_skip_downsamplings
    )

    # export model
    modalities_str = '+'.join(modalities)
    filename = f'encoder_{modalities_str}'
    filename += f'__backbone_rgbd_{type(backbone_rgbd).__name__}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = [f'input_{m}' for m in modalities]
    output_names = [f'output_{m}' for m in modalities]
    output_names += [f'skip_{ds}_{m}'
                     for ds, m in product(decoder_skip_downsamplings,
                                          modalities)]
    export_onnx_model(filepath, encoder, (encoder_input, {}),
                      input_names=input_names,
                      output_names=output_names)


@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', ('resnet18', 'resnet50', 'resnet34se'))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder1(modalities, backbone, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with same backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('backbones', (('resnet18', 'resnet34'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder2(backbones, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.xfail(reason='ONNX export is not supported yet')
@pytest.mark.parametrize('backbones', (('resnet18', 'resnet50'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder3(backbones, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',)))
@pytest.mark.parametrize('backbone', ('resnet18', 'resnet34'))
def test_encoder_rgb_or_depth_backbone(modalities, backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=None,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder_swin(modalities, backbone, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with same backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=fusion,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('backbones', (('swin-t', 'swin-t-v2'),))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder2_swin(backbones, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation='relu',
                  fusion=fusion,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',)))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2'))
def test_encoder_rgb_or_depth_backbone_swin(modalities, backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=None,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(
    EXPORT_ONNX_MODELS,
    reason='ONNX export is not supported yet'
)
@pytest.mark.parametrize('backbone', ('swin-multi-t',
                                      'swin-multi-t-v2',
                                      'swin-multi-t-128',
                                      'swin-multi-t-v2-128'))
def test_encoder_multimodal_swin(backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test2(modalities=('rgbd'),
                  backbone_rgbd=backbone,
                  normalization='layernorm',
                  tmp_path=tmp_path)



@pytest.mark.skipif(
    not IS_TIMM_AVAILABLE,
    reason="Pytorch-Image-Models is not available"
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', (
    # DINOv2
    'dinov2_small',
    'dinov2_base',
    'dinov2_large',
    'dinov2_large_reg4',
    # DINOv3
    'dinov3_small',
    'dinov3_small_qkvb',
    'dinov3_small_plus',
    'dinov3_small_plus_qkvb'
))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder_dino(modalities, backbone, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with same timm DINO backbone."""
    encoder_test1(
        modalities=modalities,
        backbone_rgb=backbone,
        backbone_depth=backbone,
        fusion=fusion,
        decoder_skip_downsamplings=(3, 6, 9, 12),
        # both ignored for DINO backbones
        activation='relu',
        normalization='layernorm',
        tmp_path=tmp_path
    )


@pytest.mark.skipif(
    not IS_TIMM_AVAILABLE,
    reason="Pytorch-Image-Models is not available"
)
@pytest.mark.parametrize('backbones', (
    # mixing different vit with add fusion is not supported
    # if models has different embed dim or number of tokens.
    ('dinov2_large_reg4', 'dinov2_large_reg4'),
    ('dinov3_small_plus', 'dinov3_small_plus')
))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder2_dino(backbones, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different timm DINO backbones."""
    encoder_test1(
        modalities=('rgb', 'depth'),
        backbone_rgb=backbones[0],
        backbone_depth=backbones[1],
        fusion=fusion,
        decoder_skip_downsamplings=(3, 6, 9, 12),
        activation='relu',
        normalization='layernorm',
        tmp_path=tmp_path
    )


@pytest.mark.skipif(
    not IS_TIMM_AVAILABLE,
    reason="Pytorch-Image-Models is not available"
)
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',)))
@pytest.mark.parametrize('backbone', (
    'dinov2_large_reg4',
    'dinov3_small_plus_qkvb'
))
def test_encoder_rgb_or_depth_backbone_dino(modalities, backbone, tmp_path):
    """Test Encoder with single modality timm DINO backbone."""
    encoder_test1(
        modalities=modalities,
        backbone_rgb=backbone,
        backbone_depth=backbone,
        fusion=None,
        decoder_skip_downsamplings=(3, 6, 9, 12),
        activation='relu',
        normalization='layernorm',
        tmp_path=tmp_path
    )

@pytest.mark.skipif(
    not IS_TIMM_AVAILABLE,
    reason="Pytorch-Image-Models is not available"
)
@pytest.mark.parametrize('backbone', ('dinov3_small_plus_qkvb',))
def test_encoder_multimodal_dino(backbone, tmp_path):
    encoder_test2(
        modalities=('rgbd',),
        backbone_rgbd=backbone,
        normalization='layernorm',  # Not used but required for swin.
        decoder_skip_downsamplings=(3, 6, 9, 12),
        tmp_path=tmp_path
    )
