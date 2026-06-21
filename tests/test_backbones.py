# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os

import pytest
import torch
from typing import Optional

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_mt_scene_analysis.model.backbone import IS_SWIN_AVAILABLE
from nicr_mt_scene_analysis.model.backbone import IS_TIMM_AVAILABLE
from nicr_mt_scene_analysis.model.backbone.pretraining import ImageNetClassifier
from nicr_mt_scene_analysis.testing import EXPORT_ONNX_MODELS
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
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason="ONNX export with dropout fails")
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
    filename += '__block_nonbottleneck1d'
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
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason="ONNX export with dropout fails")
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
    filename += '__block_nonbottleneck1d'
    filename += f'__act_{activation}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.parametrize('name', ('swin-t', 'swin-s', 'swin-b',
                                  'swin-t-v2', 'swin-s-v2', 'swin-b-v2'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (False, True))
def test_swin(name, n_input_channels, pretrained, tmp_path):
    """Test original swin transformer"""
    model = get_backbone(name,
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         # Only layernorm is supported
                         normalization='ln',
                         resnet_block=None)

    if 'v2' in name:
        x = torch.randn((2, n_input_channels, 256, 256))
    else:
        x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)

@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.parametrize('name', ('swin-t-128', 'swin-t-v2-128'))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (False, True))
def test_swin_128(name, n_input_channels, pretrained, tmp_path):
    """Test wider swin transformer"""
    if pretrained:
        # there are no pretrained weights, create them
        pretrained_filepath = os.path.join(tmp_path,
                                           f'{name}_nonbottleneck1d.pth')
        create_classifier_ckpt(pretrained_filepath, name,
                               None, n_input_channels,
                               normalization='ln')
        kwargs = {
            'pretrained_filepath': pretrained_filepath
        }
    else:
        kwargs = {}

    model = get_backbone(name,
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         # Only layernorm is supported
                         normalization='ln',
                         resnet_block=None,
                         **kwargs)

    if 'v2' in name:
        x = torch.randn((2, n_input_channels, 256, 256))
    else:
        x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.parametrize('name', ('swin-multi-t', 'swin-multi-s',
                                  'swin-multi-b', 'swin-multi-t-v2',
                                  'swin-multi-s-v2', 'swin-multi-b-v2',
                                  'swin-multi-t-128', 'swin-multi-t-v2-128'))
@pytest.mark.parametrize('n_input_channels', (4,))
@pytest.mark.parametrize('pretrained', (False, True))
def test_swin_multimodal(name, n_input_channels, pretrained, tmp_path):
    """Test modified swin transformer for multimodal data"""
    if pretrained:
        # there are no pretrained weights, create them
        pretrained_filepath = os.path.join(tmp_path,
                                           f'{name}_nonbottleneck1d.pth')
        create_classifier_ckpt(pretrained_filepath, name,
                               None, n_input_channels,
                               normalization='ln')
        kwargs = {
            'pretrained_filepath': pretrained_filepath
        }
    else:
        kwargs = {}

    model = get_backbone(name,
                         n_input_channels=n_input_channels,
                         pretrained=pretrained,
                         # Only layernorm is supported
                         normalization='ln',
                         resnet_block=None,
                         **kwargs)

    if 'v2' in name:
        x = torch.randn((2, n_input_channels, 256, 256))
    else:
        x = torch.randn((2, n_input_channels, 224, 224))
    model(x)

    filename = f'backbone_{name}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x)

@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize(
    "name",
    (
        "dinov2_small",
        "dinov2_base",
        # "dinov2_large",
        # "dinov2_giant",
        # add these only if your registry supports them:
        "dinov2_small_reg4",
        "dinov2_base_reg4",
        # "dinov2_large_reg4",
        # "dinov2_giant_reg4"
    ),
)
@pytest.mark.parametrize("n_input_channels", (1, 3))
@pytest.mark.parametrize("pretrained", (False, True))
@pytest.mark.parametrize("shape", ((256, 256), (480, 640)))
def test_timm_dinov2(name, n_input_channels, pretrained, shape, tmp_path):
    model = get_token_backbone(
        name,
        n_input_channels=n_input_channels,
        pretrained=pretrained,
    )
    model.set_input_size(shape[0], shape[1])
    assert model.backbone_meta is not None
    model.eval()
    model.model.eval()

    x = torch.randn((2, n_input_channels, shape[0], shape[1]))

    with torch.no_grad():
        # phase 1: wrapper forward matches timm forward_features
        output_nicr = model(x)
        output_timm = model.model.forward_features(x)

    assert torch.allclose(output_nicr, output_timm), \
        "Outputs of nicr and timm models do not match!"

    if not pretrained:
        return

    with torch.no_grad():
        base_tokens = model.forward_stage(0, x)  # [B, N, D]
        B, N, D = base_tokens.shape

        # create extra tokens
        # keep K modest to avoid alternate kernel / code paths with
        # different numerics.
        K = 100
        extra_tokens = torch.nn.Embedding(K, D).weight
        extra_tokens = extra_tokens.unsqueeze(0).expand(B, -1, -1)
        extra_tokens = extra_tokens.to(base_tokens.device)
        L = N + K

        n_blocks = len(model.model.blocks)

        attn_mask = torch.ones(L, L, dtype=torch.bool)
        attn_mask[:K, :] = True   # segment tokens as queries
        attn_mask[K:, :K] = False   # segment tokens as keys
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        def _forward_with_extra(mask: Optional[torch.Tensor]) -> torch.Tensor:
            tokens = model.forward_stage(
                1,
                model.forward_stage(0, x),
                extra_tokens=extra_tokens,
                attn_mask=mask
            )
            for s in range(2, n_blocks + 1):
                tokens = model.forward_stage(
                    s,
                    tokens,
                    attn_mask=mask
                )
            base, _ = model.extract_active_extra_tokens(tokens)
            model.clear_active_extra_tokens()
            return base

        output_blocked = _forward_with_extra(attn_mask)
        assert (output_nicr - output_blocked).abs().mean() < 1e-3, \
            "Blocking attention between base and extra tokens changed output!"

        output_all = _forward_with_extra(None)
        assert (output_all - output_nicr).abs().mean() > 1e-1, \
            "Allowing extra-token interaction did not change output vs " \
            "baseline!"

        assert (output_all - output_blocked).abs().mean() > 1e-1, \
             "Allowing token interaction did not change output!"


    filename = f'backbone_{name.replace("/", "_")}.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x, opset_version=18)

@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('name', (
    'dinov3_small',
    'dinov3_small_qkvb',
    'dinov3_small_plus',
    'dinov3_small_plus_qkvb',
    # 'dinov3_base',
    # 'dinov3_base_qkvb',
    # 'dinov3_large',
    # 'dinov3_large_qkvb',
    # 'dinov3_huge_plus',
    # 'dinov3_7b'
))
@pytest.mark.parametrize('n_input_channels', (1, 3))
@pytest.mark.parametrize('pretrained', (False, True))
@pytest.mark.parametrize('shape', ((256, 256), (480, 640)))
@pytest.mark.parametrize('mlp_ratio', (4, 5))
@pytest.mark.parametrize('n_heads', (12, 6))
def test_timm_dinov3(
    name,
    n_input_channels,
    pretrained,
    shape,
    mlp_ratio,
    n_heads,
    tmp_path
):

    if pretrained and mlp_ratio != 4:
        pytest.skip("pretrained weights are only available for mlp_ratio=4")

    model = get_token_backbone(
        name,
        n_input_channels=n_input_channels,
        pretrained=pretrained,
        mlp_ratio=mlp_ratio,
        num_heads=n_heads,
    )
    model.set_input_size(shape[0], shape[1])
    assert model.backbone_meta is not None
    model.eval()
    model.model.eval()

    # pretrained weights are only available for mlp_ratio=4
    for block in model.model.blocks:
        assert block.attn.num_heads == n_heads
        # SwiGLU variants use `fc1_g`, plain MLP uses `fc1`.
        fc = getattr(block.mlp, 'fc1', None) or block.mlp.fc1_g
        assert fc.out_features == fc.in_features * mlp_ratio

    x = torch.randn((2, n_input_channels, shape[0], shape[1]))

    with torch.no_grad():
        # phase 1: forward matches timm forward_features
        output_nicr = model(x)
        output_timm = model.model.forward_features(x)

    assert torch.allclose(output_nicr, output_timm), \
        "Outputs of nicr and timm models do not match!"

    # skip if pretrained=False as the effects is hard to measure without
    # meaningful weights
    if not pretrained:
        return

    with torch.no_grad():
        base_tokens = model.forward_stage(0, x)  # [B, N, D]
        B, N, D = base_tokens.shape

        # create extra tokens
        K = 100
        extra_tokens = torch.nn.Embedding(K, D).weight
        extra_tokens = extra_tokens.unsqueeze(0).expand(B, -1, -1)
        extra_tokens = extra_tokens.to(base_tokens.device)
        L = N + K

        attn_mask = torch.ones(L, L, dtype=torch.bool)
        attn_mask[:K, :] = True   # segment tokens as queries
        attn_mask[K:, :K] = False   # segment tokens as keys
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        n_blocks = len(model.model.blocks)

        def _forward_with_extra(mask: Optional[torch.Tensor]) -> torch.Tensor:
            tokens = model.forward_stage(
                1,
                model.forward_stage(0, x),
                extra_tokens=extra_tokens,
                attn_mask=mask
            )
            for s in range(2, n_blocks + 1):
                tokens = model.forward_stage(
                    s,
                    tokens,
                    attn_mask=mask
                )
            base, _ = model.extract_active_extra_tokens(tokens)
            model.clear_active_extra_tokens()
            return base

        output_blocked = _forward_with_extra(attn_mask)
        assert (output_nicr - output_blocked).abs().mean() < 1e-3, \
            "Blocking attention between base and extra tokens changed output!"

        output_all = _forward_with_extra(None)
        assert (output_all - output_nicr).abs().mean() > 1e-1, \
            "Allowing extra-token interaction did not change output vs " \
            "baseline!"

        assert (output_all - base_tokens).abs().mean() > 1e-1, \
             "Allowing token interaction did not change output!"

    filename = f'backbone_{name.replace("/", "_")}.onnx'
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, model, x, opset_version=18)
