# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

import pytest
import torch
from torch import nn

from nicr_mt_scene_analysis.model.decoder import TokenEmbeddingDecoder
from nicr_mt_scene_analysis.model.decoder import TokenMaskDecoder
from nicr_mt_scene_analysis.model.decoder import TokenOrientationDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSceneDecoder
from nicr_mt_scene_analysis.model.decoder import TokenSemanticDecoder
from nicr_mt_scene_analysis.model.encoder import ENCODER_META_KEY
from nicr_mt_scene_analysis.model import postprocessing


def _build_token_inputs(
    batch_size: int,
    n_queries: int,
    grid_size: Tuple[int, int],
    embed_dim: int,
    modalities: Tuple[str, ...] = ('rgb',)
) -> Tuple[Tuple[dict, dict, dict], dict, Tuple[str, ...]]:
    queries = {}
    x = {}
    extra_tokens = {}
    meta = {}
    n_patch_tokens = grid_size[0] * grid_size[1]

    for modality in modalities:
        q = torch.randn(batch_size, n_queries, embed_dim)
        patches = torch.randn(batch_size, n_patch_tokens, embed_dim)
        queries[modality] = q
        x[modality] = patches
        extra_tokens[modality] = q
        meta[modality] = {
            'grid_size': grid_size,
            'n_prefix_tokens': 0,
        }

    x[ENCODER_META_KEY] = meta
    return (x, {}, extra_tokens), queries, modalities


def _make_resize_meta_batch(height: int, width: int):
    return {
        '_applied_preprocessing': [[{
            'type': 'Resize',
            'valid_region_slice_y': slice(0, height),
            'valid_region_slice_x': slice(0, width),
        }]],
        'semantic_fullres': torch.zeros(1, height, width, dtype=torch.long),
        'panoptic_fullres': torch.zeros(1, height, width, dtype=torch.long),
    }


def test_token_semantic_decoder_shape():
    batch_size = 2
    n_queries = 3
    embed_dim = 8
    n_classes = 4
    grid_size = (2, 2)

    decoder = TokenSemanticDecoder(
        embed_dim=embed_dim,
        n_classes=n_classes
    )

    token_input, query_dict, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )

    logits, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    assert logits.shape == (batch_size, n_queries, n_classes)
    assert side == ()


def test_token_semantic_decoder_uses_input_queries():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    n_classes = embed_dim  # ensures linear head can mimic identity
    grid_size = (2, 2)

    decoder = TokenSemanticDecoder(
        embed_dim=embed_dim,
        n_classes=n_classes
    )
    with torch.no_grad():
        decoder.head.weight.copy_(torch.eye(embed_dim))
        decoder.head.bias.zero_()

    token_input, query_dict, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    queries = query_dict['rgb']

    logits, _ = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )
    expected = queries
    assert torch.allclose(logits, expected)


@pytest.mark.parametrize(
    'modalities, decoder_modality',
    [
        (('rgb',), 'rgb'),
        (('depth',), 'depth'),
        (('rgb', 'depth'), 'rgb'),
        (('rgb', 'depth'), 'depth'),
    ]
)
def test_token_mask_decoder_shape(modalities, decoder_modality):
    batch_size = 2
    n_queries = 2
    embed_dim = 8
    grid_size = (2, 2)

    decoder = TokenMaskDecoder(
        embed_dim=embed_dim,
        modality=decoder_modality,
        n_upsampling_blocks=2,
        upsampling_mode='bilinear',
        prediction_upsampling='bilinear'
    )

    token_input, query_dict, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim,
        modalities=modalities
    )
    queries = query_dict[decoder_modality]

    masks, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    assert masks.shape[0] == batch_size
    assert masks.shape[1] == queries.shape[1]
    out_h, out_w = masks.shape[-2:]
    assert out_h >= grid_size[0] and out_w >= grid_size[1]
    assert out_h % grid_size[0] == 0
    assert out_w % grid_size[1] == 0
    assert side == ()


def test_token_scene_decoder_uses_class_token():
    batch_size = 2
    embed_dim = 6
    n_classes = embed_dim

    decoder = TokenSceneDecoder(
        embed_dim=embed_dim,
        n_classes=n_classes
    )
    with torch.no_grad():
        decoder.head.weight.copy_(torch.eye(embed_dim))
        decoder.head.bias.zero_()

    cls_token = torch.randn(batch_size, 1, embed_dim)
    patch_tokens = torch.randn(batch_size, 4, embed_dim)
    x = {'rgb': torch.cat([cls_token, patch_tokens], dim=1)}
    extra_queries = torch.randn(batch_size, 3, embed_dim)
    token_input = (x, {}, {'rgb': extra_queries})

    logits, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    assert logits.shape == (batch_size, n_classes)
    assert side is None
    expected = cls_token[:, 0, :]
    assert torch.allclose(logits, expected)


@pytest.mark.parametrize(
    'modalities, decoder_modality',
    [
        (('rgb',), None),
        (('rgb', 'depth'), 'rgb'),
        (('rgb', 'depth'), 'depth'),
    ]
)
def test_token_scene_decoder_selects_modality(modalities, decoder_modality):
    batch_size = 1
    embed_dim = 4
    n_classes = embed_dim
    decoder_kwargs = {
        'embed_dim': embed_dim,
        'n_classes': n_classes,
    }
    if decoder_modality is not None:
        decoder_kwargs['modality'] = decoder_modality
    decoder = TokenSceneDecoder(**decoder_kwargs)
    with torch.no_grad():
        decoder.head.weight.copy_(torch.eye(embed_dim))
        decoder.head.bias.zero_()

    x = {}
    extra_tokens = {}
    cls_tokens = {}
    for idx, modality in enumerate(modalities):
        cls_token = torch.full((batch_size, 1, embed_dim), float(idx + 1))
        patches = torch.randn(batch_size, 2, embed_dim)
        x[modality] = torch.cat([cls_token, patches], dim=1)
        extra_tokens[modality] = (torch.randn(batch_size, 1, embed_dim),)
        cls_tokens[modality] = cls_token[:, 0, :]

    token_input = (x, {}, extra_tokens)
    logits, _ = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    expected_key = decoder_modality or modalities[0]
    expected = cls_tokens[expected_key]
    assert logits.shape == (batch_size, n_classes)
    assert torch.allclose(logits, expected)


def test_token_semantic_decoder_side_outputs():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    n_classes = 3
    grid_size = (2, 2)
    decoder = TokenSemanticDecoder(
        embed_dim=embed_dim,
        n_classes=n_classes
    )
    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    stage_skips = {}
    for idx in range(1, 4):
        stage_queries = torch.randn(batch_size, n_queries, embed_dim)
        stage_skips[str(idx)] = {
            '_extra_tokens': {'rgb': stage_queries}
        }
    outputs, side = decoder(
        x=token_input,
        skips=stage_skips,
        batch={},
        do_postprocessing=False
    )
    assert outputs.shape == (batch_size, n_queries, n_classes)
    assert len(side) == 3
    for tensor in side:
        assert tensor.shape == outputs.shape


def test_token_visual_embedding_decoder_side_outputs():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    grid_size = (2, 2)
    decoder = TokenEmbeddingDecoder(
        embed_dim=embed_dim,
        embedding_dim=embed_dim
    )
    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    stage_skips = {}
    for idx in range(1, 4):
        stage_queries = torch.randn(batch_size, n_queries, embed_dim)
        stage_skips[str(idx)] = {
            '_extra_tokens': {'rgb': stage_queries}
        }
    outputs, side = decoder(
        x=token_input,
        skips=stage_skips,
        batch={},
        do_postprocessing=False
    )
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(side, tuple)
    assert len(side) == 3
    for tensor in side:
        assert tensor.shape == outputs.shape


def test_token_orientation_decoder_unit_length():
    batch_size = 2
    n_queries = 3
    embed_dim = 5
    grid_size = (2, 2)

    decoder = TokenOrientationDecoder(embed_dim=embed_dim)

    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )

    orientations, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    assert orientations.shape == (batch_size, n_queries, 2)
    assert side is None
    # each query encodes a 2D biternion (sin/cos) that must stay on the
    # unit circle.
    norms = orientations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_token_orientation_decoder_with_confidence():
    batch_size = 1
    n_queries = 2
    embed_dim = 6
    grid_size = (2, 2)

    decoder = TokenOrientationDecoder(
        embed_dim=embed_dim,
        predict_confidence=True
    )
    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    outputs, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )
    assert side is None
    orientations, confidence_logits = outputs
    assert orientations.shape == (batch_size, n_queries, 2)
    assert confidence_logits.shape == (batch_size, n_queries)


def test_token_orientation_postprocessing_adds_angles():
    orientations = torch.randn(1, 2, 2)
    post = postprocessing.TokenOrientationPostprocessing()
    result = post.postprocess((orientations, None), batch={}, is_training=False)
    assert 'token_orientation_angles' in result
    angles = result['token_orientation_angles']
    assert angles.shape == (1, 2)


def test_token_embedding_decoder_linear_head():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    grid_size = (2, 2)

    decoder = TokenEmbeddingDecoder(
        embed_dim=embed_dim,
        embedding_dim=embed_dim
    )
    # the embedding head is an nn.Sequential MLP. Exercise it via its
    # constituent nn.Linear layers.
    linear_layers = [m for m in decoder.head if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2
    with torch.no_grad():
        for layer in linear_layers:
            layer.weight.zero_()
            layer.bias.zero_()

    token_input, query_dict, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )

    embeddings, side = decoder(
        x=token_input,
        skips={},
        batch={},
        do_postprocessing=False
    )

    assert embeddings.shape == (batch_size, n_queries, embed_dim)
    assert side == ()
    # with all-zero head weights/biases the MLP output is the zero embedding
    assert torch.allclose(embeddings, torch.zeros_like(embeddings))


def test_token_mask_postprocessing_stores_probs():
    logits = torch.zeros(1, 2, 2, 2)
    post = postprocessing.TokenMaskPostprocessing()
    result = post.postprocess((logits, ()), batch={}, is_training=False)
    assert torch.equal(result['token_mask_output'], logits)
    assert torch.equal(result['token_mask_probs'], torch.sigmoid(logits))
    assert result['token_mask_side_outputs'] == ()


def test_token_mask_decoder_multistage_outputs():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    grid_size = (2, 2)
    decoder = TokenMaskDecoder(
        embed_dim=embed_dim,
        n_upsampling_blocks=2,
        upsampling_mode='bilinear',
        prediction_upsampling='bilinear'
    )
    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    stage_skips = {}
    for idx in range(1, 4):
        stage_queries = torch.randn(batch_size, n_queries, embed_dim)
        stage_tokens = torch.randn(
            batch_size,
            grid_size[0] * grid_size[1],
            embed_dim
        )
        stage_skips[str(idx)] = {
            'rgb': stage_tokens,
            '_extra_tokens': {'rgb': stage_queries}
        }
    decoder.train()
    outputs, side = decoder(
        x=token_input,
        skips=stage_skips,
        batch={},
        do_postprocessing=False
    )
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(side, tuple)
    assert len(side) == 3
    for tensor in side:
        assert tensor.shape == outputs.shape


def test_token_mask_decoder_multistage_inference_uses_final_stage():
    batch_size = 1
    n_queries = 2
    embed_dim = 4
    grid_size = (2, 2)
    decoder = TokenMaskDecoder(
        embed_dim=embed_dim,
        n_upsampling_blocks=2,
        upsampling_mode='bilinear',
        prediction_upsampling='bilinear'
    )
    token_input, _, _ = _build_token_inputs(
        batch_size,
        n_queries,
        grid_size,
        embed_dim
    )
    stage_skips = {}
    for idx in range(1, 4):
        stage_queries = torch.randn(batch_size, n_queries, embed_dim)
        stage_tokens = torch.randn(
            batch_size,
            grid_size[0] * grid_size[1],
            embed_dim
        )
        stage_skips[str(idx)] = {
            'rgb': stage_tokens,
            '_extra_tokens': {'rgb': stage_queries}
        }
    decoder.eval()
    outputs, side = decoder(
        x=token_input,
        skips=stage_skips,
        batch={},
        do_postprocessing=False
    )
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(side, tuple)
    assert len(side) == 3
    for tensor in side:
        assert tensor.shape == outputs.shape


def test_token_semantic_postprocessing_requires_mask_probs():
    post = postprocessing.TokenSemanticPostprocessing()
    logits = torch.randn(1, 1, 2)
    with pytest.raises(KeyError):
        post.postprocess(
            (logits, ()),
            batch={},
            is_training=False
        )


def test_token_semantic_postprocessing_builds_dense_predictions():
    mask_logits = torch.tensor([[[[20.0, -20.0]], [[-20.0, 20.0]]]])
    mask_post = postprocessing.TokenMaskPostprocessing()
    batch = _make_resize_meta_batch(height=1, width=2)
    mask_outputs = mask_post.postprocess(
        (mask_logits, None), batch=batch, is_training=False
    )
    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    post = postprocessing.TokenSemanticPostprocessing()
    result = post.postprocess(
        (logits, ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    dense_idx = result['token_semantic_dense_idx']
    assert dense_idx.shape == (1, 1, 2)
    # left pixel -> query 0 -> class 1, right pixel -> query 1 -> class 0
    expected = torch.tensor([[1, 0]])
    assert torch.equal(dense_idx[0], expected)


def test_token_panoptic_postprocessing_uses_shared_context():
    # strongly polarised logits so each query mask covers exactly one pixel
    mask_logits = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]])
    mask_post = postprocessing.TokenMaskPostprocessing()
    batch = _make_resize_meta_batch(height=1, width=2)
    mask_outputs = mask_post.postprocess(
        (mask_logits, None), batch=batch, is_training=False
    )
    class_logits = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])
    semantic_post = postprocessing.TokenSemanticPostprocessing()
    semantic_outputs = semantic_post.postprocess(
        (class_logits, ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    post = postprocessing.TokenPanopticPostprocessing(
        semantic_classes_is_thing=(False, True, True),
        mask_threshold=0.5,
        overlap_threshold=1.0,
        max_instances_per_category=10
    )
    # panoptic postprocessing reuses the shared semantic context in out_dict.
    # the semantic logits are passed through data as ``(logits, side)``.
    result = post.postprocess(
        (class_logits, ()),
        batch=batch,
        is_training=False,
        out_dict=semantic_outputs
    )
    seg = result['token_panoptic_segmentation']
    assert seg.shape == (1, 1, 2)
    # left pixel -> class 1 -> id 20, right pixel -> class 2 -> id 31
    assert seg[0, 0, 0] == 20
    assert seg[0, 0, 1] == 31
