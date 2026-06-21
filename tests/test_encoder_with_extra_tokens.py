# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict
from typing import Sequence
from typing import Tuple

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_token_backbone
from nicr_mt_scene_analysis.model.backbone import IS_TIMM_AVAILABLE
from nicr_mt_scene_analysis.model.encoder import ENCODER_META_KEY
from nicr_mt_scene_analysis.model.encoder import get_encoder
from nicr_mt_scene_analysis.model.token_encoder import get_token_encoder


_BASE_SKIP_DOWNSAMPLINGS = (3, 6, 9, 12)
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_BATCH_SIZE = 1
_N_EXTRA_TOKENS = 8


def _build_dino_backbone(name: str, n_input_channels: int):
    backbone = get_token_backbone(
        name=name,
        n_input_channels=n_input_channels,
        pretrained=False,
    )
    backbone.set_input_size(height=_IMAGE_HEIGHT, width=_IMAGE_WIDTH)
    return backbone


def _resolve_skip_downsamplings(
    stages_downsampling: Sequence[int]
) -> Tuple[int, ...]:
    max_stage = stages_downsampling[-1]
    valid = tuple(ds for ds in _BASE_SKIP_DOWNSAMPLINGS if ds <= max_stage)
    if not valid:
        valid = (max_stage,)
    return valid


def _make_encoder_input(modality: str) -> torch.Tensor:
    channels = 3 if modality == 'rgb' else 1
    return torch.randn(_BATCH_SIZE, channels, _IMAGE_HEIGHT, _IMAGE_WIDTH)


def _make_extra_tokens(embed_dim: int) -> torch.Tensor:
    return torch.randn(_BATCH_SIZE, _N_EXTRA_TOKENS, embed_dim)


def _assert_skip_tokens(
    skips: Dict[str, Dict[str, torch.Tensor]],
    skip_downsamplings: Sequence[int],
    expected_modalities: Tuple[str, ...],
    start_idx: int,
    end_idx: int
) -> None:
    active = {ds for ds in skip_downsamplings if start_idx <= ds <= end_idx}
    for ds in skip_downsamplings:
        ds_key = str(ds)
        skip_entry = skips[ds_key]
        extra_tokens = skip_entry.get('_extra_tokens')
        if ds in active and expected_modalities:
            assert extra_tokens is not None
            prefix_tokens = extra_tokens.get('_prefix_tokens', {})
            token_entries = {
                k: v for k, v in extra_tokens.items()
                if k != '_prefix_tokens'
            }
            assert set(token_entries.keys()) == set(expected_modalities)
            for modality in expected_modalities:
                tensor = token_entries[modality]
                assert isinstance(tensor, torch.Tensor)
                assert tensor.shape[0] == _BATCH_SIZE
                assert tensor.shape[1] == _N_EXTRA_TOKENS
            if prefix_tokens:
                assert isinstance(prefix_tokens, dict)
        else:
            assert '_extra_tokens' not in skip_entry


def _assert_extra_tokens(
    extra_tokens: Dict[str, torch.Tensor],
    expected_modalities: Tuple[str, ...],
    embed_dims: Dict[str, int]
) -> None:
    prefix_tokens = extra_tokens.get('_prefix_tokens', {})
    token_entries = {
        k: v for k, v in extra_tokens.items()
        if k != '_prefix_tokens'
    }
    assert set(token_entries.keys()) == set(expected_modalities)
    for modality in expected_modalities:
        tensor = token_entries[modality]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == _BATCH_SIZE
        assert tensor.shape[1] == _N_EXTRA_TOKENS
        assert tensor.shape[2] == embed_dims[modality]
    if prefix_tokens:
        assert isinstance(prefix_tokens, dict)

@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
@pytest.mark.parametrize('modality', ('rgb', 'depth'))
def test_encoder_with_extra_tokens_single_modality(backbone_name, modality):
    backbone = _build_dino_backbone(
        name=backbone_name,
        n_input_channels=3 if modality == 'rgb' else 1
    )
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone.stages_downsampling
    )
    start_idx = skip_downsamplings[0]
    end_idx = start_idx

    encoder = get_token_encoder(
        backbone_rgb=backbone if modality == 'rgb' else None,
        backbone_depth=backbone if modality == 'depth' else None,
        backbone_rgbd=None,
        fusion=None,
        normalization='layernorm',
        activation='relu',
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=start_idx,
        extra_tokens_end_stage_idx=end_idx
    )

    extra_tokens = {modality: _make_extra_tokens(backbone.embed_dim)}
    x = {modality: _make_encoder_input(modality)}

    with torch.no_grad():
        x, skips, extra_tokens_out = encoder(x, extra_tokens=extra_tokens)

    feature_modalities = sorted(k for k in x.keys()
                                if k != ENCODER_META_KEY)
    assert feature_modalities == [modality]
    assert ENCODER_META_KEY in x
    assert modality in x[ENCODER_META_KEY]
    modality_meta = x[ENCODER_META_KEY][modality]
    assert 'grid_size' in modality_meta
    assert 'patch_size' in modality_meta
    assert set(skips.keys()) == {str(ds) for ds in skip_downsamplings}
    _assert_skip_tokens(
        skips=skips,
        skip_downsamplings=skip_downsamplings,
        expected_modalities=(modality,),
        start_idx=start_idx,
        end_idx=end_idx
    )
    _assert_extra_tokens(
        extra_tokens=extra_tokens_out,
        expected_modalities=(modality,),
        embed_dims={modality: backbone.embed_dim}
    )


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
@pytest.mark.parametrize('backbone_name', ('dinov2_small', 'dinov3_small_plus'))
@pytest.mark.parametrize('token_modalities', (
    ('rgb',),
    ('depth',),
    ('rgb', 'depth')
))
def test_fused_encoder_with_extra_tokens(backbone_name, token_modalities):
    backbone_rgb = _build_dino_backbone(name=backbone_name, n_input_channels=3)
    backbone_depth = _build_dino_backbone(
        name=backbone_name,
        n_input_channels=1
    )
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone_rgb.stages_downsampling
    )
    start_idx = skip_downsamplings[0]
    fusion_stage_indices = (skip_downsamplings[-1],)
    if len(skip_downsamplings) > 1:
        end_idx = skip_downsamplings[-2]
    else:
        end_idx = fusion_stage_indices[0]

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

    extra_tokens = {}
    if 'rgb' in token_modalities:
        extra_tokens['rgb'] = _make_extra_tokens(backbone_rgb.embed_dim)
    if 'depth' in token_modalities:
        extra_tokens['depth'] = _make_extra_tokens(backbone_depth.embed_dim)

    x = {
        'rgb': _make_encoder_input('rgb'),
        'depth': _make_encoder_input('depth')
    }

    with torch.no_grad():
        x, skips, extra_tokens_out = encoder(x, extra_tokens=extra_tokens)

    feature_modalities = sorted(k for k in x.keys()
                                if k != ENCODER_META_KEY)
    assert feature_modalities == ['depth', 'rgb']
    assert ENCODER_META_KEY in x
    assert set(x[ENCODER_META_KEY].keys()) == {'rgb', 'depth'}
    assert set(skips.keys()) == {str(ds) for ds in skip_downsamplings}
    _assert_skip_tokens(
        skips=skips,
        skip_downsamplings=skip_downsamplings,
        expected_modalities=token_modalities,
        start_idx=start_idx,
        end_idx=end_idx
    )

    expected_dims = {
        'rgb': backbone_rgb.embed_dim,
        'depth': backbone_depth.embed_dim
    }
    _assert_extra_tokens(
        extra_tokens=extra_tokens_out,
        expected_modalities=token_modalities,
        embed_dims=expected_dims
    )


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
def test_encoder_with_extra_tokens_modify_features_without_mask():
    backbone_plain = _build_dino_backbone(
        name='dinov2_small',
        n_input_channels=3
    )
    backbone_tokens = _build_dino_backbone(
        name='dinov2_small',
        n_input_channels=3
    )
    n_stages = len(backbone_plain.stages)
    start_idx = 1
    end_idx = n_stages - 1

    skip_downsamplings = (3, 6, 9, 12)
    encoder_plain = get_encoder(
        backbone_rgb=backbone_plain,
        backbone_depth=None,
        backbone_rgbd=None,
        fusion=None,
        normalization='layernorm',
        activation='relu',
        skip_downsamplings=skip_downsamplings
    )
    encoder_tokens = get_token_encoder(
        backbone_rgb=backbone_tokens,
        backbone_depth=None,
        backbone_rgbd=None,
        fusion=None,
        normalization='layernorm',
        activation='relu',
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=start_idx,
        extra_tokens_end_stage_idx=end_idx
    )

    x = {'rgb': _make_encoder_input('rgb')}
    extra_tokens = {'rgb': _make_extra_tokens(backbone_tokens.embed_dim)}

    with torch.no_grad():
        x_plain, _ = encoder_plain(x)
        x_tokens, _, _ = encoder_tokens(x, extra_tokens=extra_tokens)

    diff = (x_tokens['rgb'] - x_plain['rgb']).abs().mean()
    assert diff > 1e-1


@pytest.mark.skipif(not IS_TIMM_AVAILABLE,
                   reason="Pytorch-Image-Models is not available")
def test_encoder_with_extra_tokens_attention_mask_blocks_interaction():
    backbone = _build_dino_backbone(name='dinov2_small', n_input_channels=3)
    skip_downsamplings = _resolve_skip_downsamplings(
        backbone.stages_downsampling
    )
    start_idx = skip_downsamplings[0]
    end_idx = start_idx

    encoder = get_token_encoder(
        backbone_rgb=backbone,
        backbone_depth=None,
        backbone_rgbd=None,
        fusion=None,
        normalization='layernorm',
        activation='relu',
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=start_idx,
        extra_tokens_end_stage_idx=end_idx
    )

    x = {'rgb': _make_encoder_input('rgb')}
    extra_tokens = {'rgb': _make_extra_tokens(backbone.embed_dim)}

    def _run_and_record(**forward_kwargs):
        block_idx = start_idx - 1
        block = backbone.model.blocks[block_idx]
        records = []
        lengths = []

        def _record_pre_hook(module, inputs, kwargs):
            mask = kwargs.get('attn_mask')
            records.append(None if mask is None else mask.detach().clone())
            if inputs:
                lengths.append(inputs[0].shape[1])
            else:
                lengths.append(None)
            # no exception. let the forward finish so hooks run cleanly.

        handle = block.register_forward_pre_hook(
            _record_pre_hook,
            with_kwargs=True
        )
        encoder(
            x,
            extra_tokens=extra_tokens,
            **forward_kwargs
        )
        handle.remove()
        return records, lengths

    def _build_mask(total_tokens: int) -> torch.Tensor:
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)
        mask[:_N_EXTRA_TOKENS, :] = True
        mask[_N_EXTRA_TOKENS:, :_N_EXTRA_TOKENS] = False
        return mask.unsqueeze(0).unsqueeze(0)

    captured_masks = []
    def _build_controller(mask_tensor: torch.Tensor):
        class _DummyController:
            def should_apply(self, modality, stage_idx):
                return True

            def build_mask(
                self,
                *,
                queries,
                patch_tokens,
                grid_size,
                n_prefix_tokens,
                modality,
                stage_idx,
                **_
            ):
                    captured_masks.append(mask_tensor.clone())
                    return mask_tensor.to(queries.device)

        return _DummyController()

    with torch.no_grad():
        recorded_no_mask, lengths_no_mask = _run_and_record()
        total_tokens = lengths_no_mask[0]
        mask = _build_mask(total_tokens)
        controller = _build_controller(mask)
        recorded_with_mask, _ = _run_and_record(
            extra_tokens_controller=controller
        )

    assert recorded_no_mask == [None]
    assert len(recorded_with_mask) == 1
    assert recorded_with_mask[0] is not None
    assert captured_masks, "No mask captured during controller hook."
