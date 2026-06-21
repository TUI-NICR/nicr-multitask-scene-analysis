# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, Tuple

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_mt_scene_analysis.model.backbone import IS_TIMM_AVAILABLE
from nicr_mt_scene_analysis.model.decoder import TokenEmbeddingDecoder
from nicr_mt_scene_analysis.model.decoder import TokenMaskDecoder
from nicr_mt_scene_analysis.model.decoder import TokenOrientationDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSceneDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSemanticDecoder
from nicr_mt_scene_analysis.model.encoder import ENCODER_META_KEY
from nicr_mt_scene_analysis.model.token_encoder import get_token_encoder


_IMAGE_HEIGHT = 480
_IMAGE_WIDTH = 640
_BATCH_SIZE = 1
_N_EXTRA_TOKENS = 100
_BASE_SKIP_DOWNSAMPLINGS = (3, 6, 9, 12)


def _build_dino_backbone(name: str, n_input_channels: int):
    backbone = get_token_backbone(
        name=name,
        n_input_channels=n_input_channels,
        pretrained=False,
    )
    backbone.set_input_size(height=_IMAGE_HEIGHT, width=_IMAGE_WIDTH)
    return backbone


def _resolve_skip_downsamplings(stages_downsampling) -> Tuple[int, ...]:
    max_stage = stages_downsampling[-1]
    valid = tuple(ds for ds in _BASE_SKIP_DOWNSAMPLINGS if ds <= max_stage)
    if not valid:
        valid = (max_stage,)
    return valid


def _make_extra_tokens(embed_dim: int) -> torch.Tensor:
    return torch.randn(_BATCH_SIZE, _N_EXTRA_TOKENS, embed_dim)


def _assert_encoder_meta(x: Dict[str, torch.Tensor],
                         expected_modalities: Tuple[str, ...]) -> None:
    assert ENCODER_META_KEY in x
    meta = x[ENCODER_META_KEY]
    assert set(meta.keys()) == set(expected_modalities)
    for modality in expected_modalities:
        modality_meta = meta[modality]
        grid_h, grid_w = modality_meta['grid_size']
        prefix = modality_meta.get('n_prefix_tokens', 0)
        tokens = x.get(modality)
        if tokens is not None:
            assert tokens.shape[1] == prefix + grid_h * grid_w


def _run_decoders(
    token_input,
    skips,
    *,
    embed_dim: int,
    modality: str
) -> None:
    features_only, decoder_skips, extra_tokens = token_input
    meta = features_only.get(ENCODER_META_KEY, {})
    modality_meta = meta.get(modality)
    if not modality_meta:
        raise AssertionError(f'No encoder meta for modality {modality}.')
    decoders = [
        TokenSemanticDecoder(
            embed_dim=embed_dim,
            n_classes=5,
            modality=modality
        ),
        TokenSceneDecoder(embed_dim=embed_dim, n_classes=3, modality=modality),
        TokenMaskDecoder(embed_dim=embed_dim,
                         modality=modality),
        TokenEmbeddingDecoder(embed_dim=embed_dim,
                              embedding_dim=embed_dim // 2,
                              modality=modality),
        TokenOrientationDecoder(embed_dim=embed_dim, modality=modality),
    ]
    for decoder in decoders:
        current_input = token_input
        out, side = decoder(
            x=current_input,
            skips=skips,
            batch={},
            do_postprocessing=False
        )
        assert out is not None
        if isinstance(decoder, (TokenOrientationDecoder, TokenSceneDecoder)):
            assert side is None
            continue
        assert isinstance(side, tuple)
        for aux in side:
            assert aux.shape[0] == out.shape[0]


def _forward_token_encoder(encoder, inputs, extra_tokens):
    with torch.no_grad():
        outputs = encoder(inputs, extra_tokens=extra_tokens)
    assert len(outputs) == 3
    return outputs


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
def test_token_pipeline_single_rgbd_encoder(backbone_name):
    backbone_rgbd = _build_dino_backbone(backbone_name, n_input_channels=4)
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone_rgbd.stages_downsampling
    )
    start_idx = skip_downsamplings[0]
    end_idx = skip_downsamplings[-1]
    encoder = get_token_encoder(
        backbone_rgb=None,
        backbone_depth=None,
        backbone_rgbd=backbone_rgbd,
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=start_idx,
        extra_tokens_end_stage_idx=end_idx
    )

    modality = 'rgbd'
    inputs = {
        modality: torch.randn(_BATCH_SIZE, 4, _IMAGE_HEIGHT, _IMAGE_WIDTH)
    }
    extra_tokens = {
        modality: _make_extra_tokens(backbone_rgbd.embed_dim)
    }

    features_out, skips, extra_tokens_out = _forward_token_encoder(
        encoder,
        inputs,
        extra_tokens
    )
    assert modality in features_out
    assert modality in extra_tokens_out
    _assert_encoder_meta(features_out, (modality,))
    token_input = (features_out, skips, extra_tokens_out)
    _run_decoders(token_input,
                  skips,
                  embed_dim=backbone_rgbd.embed_dim,
                  modality=modality)


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
def test_token_pipeline_fused_rgb_depth_encoder(backbone_name):
    backbone_rgb = _build_dino_backbone(backbone_name, n_input_channels=3)
    backbone_depth = _build_dino_backbone(backbone_name, n_input_channels=1)

    skip_downsamplings = _resolve_skip_downsamplings(
        backbone_rgb.stages_downsampling
    )
    start_idx = skip_downsamplings[0]
    end_idx = skip_downsamplings[-1]
    fusion_stage_indices = (end_idx,)

    encoder = get_token_encoder(
        backbone_rgb=backbone_rgb,
        backbone_depth=backbone_depth,
        backbone_rgbd=None,
        fusion='add-uni-rgb',
        normalization='layernorm',
        activation='relu',
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=start_idx,
        extra_tokens_end_stage_idx=end_idx,
        fusion_stage_indices=fusion_stage_indices
    )

    inputs = {
        'rgb': torch.randn(_BATCH_SIZE, 3, _IMAGE_HEIGHT, _IMAGE_WIDTH),
        'depth': torch.randn(_BATCH_SIZE, 1, _IMAGE_HEIGHT, _IMAGE_WIDTH),
    }
    extra_tokens = {
        'rgb': _make_extra_tokens(backbone_rgb.embed_dim),
        'depth': _make_extra_tokens(backbone_depth.embed_dim),
    }

    features_out, skips, extra_tokens_out = _forward_token_encoder(
        encoder,
        inputs,
        extra_tokens
    )
    feature_modalities = {
        k for k in features_out.keys()
        if k != ENCODER_META_KEY
    }
    assert feature_modalities == {'rgb', 'depth'}
    extra_token_modalities = {
        k for k in extra_tokens_out.keys()
        if k != '_prefix_tokens'
    }
    assert extra_token_modalities == {'rgb', 'depth'}
    _assert_encoder_meta(features_out, ('rgb', 'depth'))

    token_input = (features_out, skips, extra_tokens_out)
    for modality in ('rgb', 'depth'):
        if modality == 'rgb':
            embed_dim = backbone_rgb.embed_dim
        else:
            embed_dim = backbone_depth.embed_dim
        _run_decoders(token_input,
                      skips,
                      embed_dim=embed_dim,
                      modality=modality)


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
def test_token_encoder_metadata_without_extra_tokens_single(backbone_name):
    backbone_rgbd = _build_dino_backbone(backbone_name, n_input_channels=4)
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone_rgbd.stages_downsampling
    )
    encoder = get_token_encoder(
        backbone_rgb=None,
        backbone_depth=None,
        backbone_rgbd=backbone_rgbd,
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=skip_downsamplings[0],
        extra_tokens_end_stage_idx=skip_downsamplings[-1]
    )
    x, skips = encoder({
        'rgbd': torch.randn(_BATCH_SIZE, 4, _IMAGE_HEIGHT, _IMAGE_WIDTH)
    })
    _assert_encoder_meta(x, ('rgbd',))
    token_input = (x, skips, {})
    with pytest.raises(ValueError):
        _run_decoders(token_input,
                      skips,
                      embed_dim=backbone_rgbd.embed_dim,
                      modality='rgbd')


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
def test_token_encoder_partial_extra_tokens(backbone_name):
    backbone_rgb = _build_dino_backbone(backbone_name, n_input_channels=3)
    backbone_depth = _build_dino_backbone(backbone_name, n_input_channels=1)
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone_rgb.stages_downsampling
    )
    encoder = get_token_encoder(
        backbone_rgb=backbone_rgb,
        backbone_depth=backbone_depth,
        fusion='se-add-uni-rgb',
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=skip_downsamplings[0],
        extra_tokens_end_stage_idx=skip_downsamplings[-1],
        fusion_stage_indices=tuple()
    )
    inputs = {
        'rgb': torch.randn(_BATCH_SIZE, 3, _IMAGE_HEIGHT, _IMAGE_WIDTH),
        'depth': torch.randn(_BATCH_SIZE, 1, _IMAGE_HEIGHT, _IMAGE_WIDTH),
    }
    extra_tokens = {
        'rgb': _make_extra_tokens(backbone_rgb.embed_dim)
    }
    features_out, skips, extra_tokens_out = _forward_token_encoder(
        encoder,
        inputs,
        extra_tokens
    )
    extra_token_modalities = {
        k for k in extra_tokens_out.keys()
        if k != '_prefix_tokens'
    }
    assert extra_token_modalities == {'rgb'}
    _assert_encoder_meta(features_out, ('rgb', 'depth'))
    token_input = (features_out, skips, extra_tokens_out)
    _run_decoders(token_input,
                  skips,
                  embed_dim=backbone_rgb.embed_dim,
                  modality='rgb')
    with pytest.raises(ValueError):
        _run_decoders(token_input,
                      skips,
                      embed_dim=backbone_depth.embed_dim,
                      modality='depth')
