# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import pytest
import torch

from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres_key
from nicr_mt_scene_analysis.model import postprocessing
from nicr_mt_scene_analysis.task_helper import IS_TRANSFORMERS_AVAILABLE
from nicr_mt_scene_analysis.task_helper import TokenMaskTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenSemanticTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenPanopticTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenOrientationTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenEmbeddingTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenSceneTaskHelper
from nicr_mt_scene_analysis.task_helper import TokenMatchingCache


def test_token_mask_postprocessing_adds_probs():
    logits = torch.randn(1, 2, 4, 4)
    post = postprocessing.TokenMaskPostprocessing()
    training = post.postprocess((logits, ()), batch={}, is_training=True)
    assert torch.equal(training['token_mask_output'], logits)
    assert training['token_mask_side_outputs'] == ()
    inference = post.postprocess((logits, ()), batch={}, is_training=False)
    assert 'token_mask_probs' in inference
    assert torch.equal(inference['token_mask_output'], logits)
    assert inference['token_mask_side_outputs'] == ()


def test_token_semantic_postprocessing_adds_probabilities():
    logits = torch.randn(1, 3, 5)
    post = postprocessing.TokenSemanticPostprocessing()
    mask_post = postprocessing.TokenMaskPostprocessing()
    mask_logits = torch.zeros(1, logits.shape[1], 1, 1)
    batch = _build_resize_meta_batch(height=1, width=1)
    mask_outputs = mask_post.postprocess(
        (mask_logits, ()), batch=batch, is_training=False
    )
    inference = post.postprocess(
        (logits, ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    assert 'token_semantic_softmax_scores' in inference
    assert 'token_semantic_segmentation_idx' in inference
    assert inference['token_semantic_softmax_scores'].shape == logits.shape
    assert 'token_semantic_dense_output' in inference
    assert 'token_semantic_side_outputs' in inference


def _build_resize_meta_batch(height: int, width: int):
    resize_meta = [{
        'type': 'Resize',
        'valid_region_slice_y': slice(0, height),
        'valid_region_slice_x': slice(0, width),
    }]
    return {
        '_applied_preprocessing': [resize_meta],
        'semantic_fullres': torch.zeros(1, height, width, dtype=torch.long),
        'panoptic_fullres': torch.zeros(1, height, width, dtype=torch.long),
    }


def _build_batch_and_predictions():
    # create a simple semantic batch (single query + mask)
    mask_logits = torch.zeros(1, 2, 2, 2)
    mask_logits[0, 0] = 5.0
    class_logits = torch.zeros(1, 2, 2)
    class_logits[0, 0, 0] = 8.0    # query 0 -> class 0 (dataset label 1)
    class_logits[0, 1, 1] = 8.0    # unmatched query still favors class 1
    class_idx = torch.argmax(class_logits, dim=-1)
    semantic_tensor = torch.zeros(1, 2, 2, dtype=torch.long)
    batch = _build_resize_meta_batch(height=2, width=2)
    batch.update({
        'semantic_token_masks': [torch.ones(1, 2, 2)],
        'semantic_token_labels': [torch.tensor([1])],
        'semantic': semantic_tensor,
        'semantic_fullres': semantic_tensor.clone(),
        'token_mask_output': mask_logits,
        'token_semantic_output': class_logits
    })
    predictions_post = {
        'token_mask_output': mask_logits,
        'token_mask_side_outputs': (),
        'token_semantic_output': class_logits,
        'token_semantic_side_outputs': (),
        'token_semantic_segmentation_idx': class_idx,
    }
    mask_post = postprocessing.TokenMaskPostprocessing()
    mask_outputs = mask_post.postprocess(
        (mask_logits, ()),
        batch=batch,
        is_training=False
    )
    semantic_post = postprocessing.TokenSemanticPostprocessing()
    semantic_outputs = semantic_post.postprocess(
        (class_logits, ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    predictions_post.update(mask_outputs)
    predictions_post.update(semantic_outputs)
    return batch, predictions_post


def _build_panoptic_batch_and_predictions():
    # synthetic panoptic batch containing two instances. Off-region logits
    # are strongly negative so the sigmoid mask probabilities are clearly
    # foreground/background.
    mask_logits = torch.tensor([[
        [[10, 10], [-10, -10]],
        [[-10, -10], [10, 10]],
    ]], dtype=torch.float32)
    class_logits = torch.tensor([[
        [0.0, 10.0, -1.0],
        [0.0, -5.0, 10.0],
    ]], dtype=torch.float32)
    class_idx = torch.argmax(class_logits, dim=-1)
    panoptic_masks = torch.tensor([[
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]], dtype=torch.float32)
    panoptic_labels = torch.tensor([1, 2], dtype=torch.long)
    max_instances = 1 << 16
    panoptic_tensor = torch.tensor([[
        [1 * max_instances + 1, 1 * max_instances + 1],
        [2 * max_instances + 1, 2 * max_instances + 1]
    ]])
    semantic_tensor = torch.tensor([[
        [1, 1],
        [2, 2]
    ]])
    batch = _build_resize_meta_batch(height=2, width=2)
    batch.update({
        'panoptic_token_masks': [panoptic_masks.squeeze(0)],
        'panoptic_token_labels': [panoptic_labels],
        'panoptic': panoptic_tensor,
        'panoptic_fullres': panoptic_tensor.clone(),
        'panoptic_ids_to_instance_dict': [{
            1 * max_instances + 1: 1,
            2 * max_instances + 1: 1
        }],
        'semantic': semantic_tensor,
        'semantic_fullres': semantic_tensor.clone(),
        'token_mask_output': mask_logits,
        'token_semantic_output': class_logits
    })
    predictions_post = {
        'token_mask_output': mask_logits,
        'token_mask_side_outputs': (),
        'token_semantic_output': class_logits,
        'token_semantic_side_outputs': (),
        'token_semantic_segmentation_idx': class_idx,
    }
    return batch, predictions_post


def _attach_panoptic_predictions(
    predictions_post,
    *,
    semantic_classes_is_thing
):
    # run panoptic postprocessing once to mimic inference pipeline
    mask_logits = predictions_post['token_mask_output']
    height, width = mask_logits.shape[-2:]
    batch = _build_resize_meta_batch(height=height, width=width)
    mask_post = postprocessing.TokenMaskPostprocessing()
    mask_outputs = mask_post.postprocess(
        (mask_logits, ()),
        batch=batch,
        is_training=False
    )
    semantic_post = postprocessing.TokenSemanticPostprocessing()
    semantic_outputs = semantic_post.postprocess(
        (predictions_post['token_semantic_output'], ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    post = postprocessing.TokenPanopticPostprocessing(
        semantic_classes_is_thing=semantic_classes_is_thing
    )
    outputs = post.postprocess(
        data=(predictions_post['token_semantic_output'], ()),
        batch=batch,
        is_training=False,
        out_dict=semantic_outputs
    )
    predictions_post.update(outputs)



@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_mask_helper_and_semantic_helper_share_matching():
    torch.manual_seed(0)
    cache = TokenMatchingCache()
    mask_helper = TokenMaskTaskHelper(
        mask_mode='semantic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='semantic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    semantic_helper = TokenSemanticTaskHelper(
        n_classes=2, matching_cache=cache
    )
    device = torch.device('cpu')
    mask_helper.initialize(device)
    semantic_helper.initialize(device)
    mask_helper.train()
    semantic_helper.train()
    batch, predictions_post = _build_batch_and_predictions()

    losses_mask, _ = mask_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_semantic_mask_total_loss' in losses_mask
    result = cache[0]
    assert len(result) == 1
    assert torch.equal(result[0][0], torch.tensor([0]))

    losses_sem, _ = semantic_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_semantic_total_loss' in losses_sem


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_mask_helper_uses_side_outputs():
    torch.manual_seed(0)
    cache = TokenMatchingCache()
    helper = TokenMaskTaskHelper(
        mask_mode='semantic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='semantic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch, predictions_post = _build_batch_and_predictions()
    logits = predictions_post['token_mask_output']
    predictions_with_aux = dict(predictions_post)
    predictions_with_aux['token_mask_side_outputs'] = (logits,)
    predictions_with_aux['token_semantic_side_outputs'] = (
        predictions_post['token_semantic_output'],
    )
    losses_aux, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_with_aux
    )
    losses_base, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_semantic_loss_mask_aux_1' in losses_aux
    assert 'token_semantic_loss_dice_aux_1' in losses_aux
    expected_mask = (
        losses_aux['token_semantic_loss_mask_main']
        + losses_aux['token_semantic_loss_mask_aux_1']
    )
    expected_dice = (
        losses_aux['token_semantic_loss_dice_main']
        + losses_aux['token_semantic_loss_dice_aux_1']
    )
    assert torch.allclose(losses_aux['token_semantic_loss_mask'], expected_mask)
    assert torch.allclose(losses_aux['token_semantic_loss_dice'], expected_dice)
    expected_total = expected_mask + expected_dice
    if 'token_semantic_loss_cls_main' in losses_aux:
        expected_cls = (
            losses_aux['token_semantic_loss_cls_main']
            + losses_aux['token_semantic_loss_cls_aux_1']
        )
        assert torch.allclose(
            losses_aux['token_semantic_loss_cls'], expected_cls
        )
        expected_total += expected_cls
    assert torch.allclose(
        losses_aux['token_semantic_mask_total_loss'], expected_total
    )
    assert torch.allclose(
        losses_base['token_semantic_loss_mask'],
        losses_base['token_semantic_loss_mask_main']
    )
    assert torch.allclose(
        losses_base['token_semantic_loss_dice'],
        losses_base['token_semantic_loss_dice_main']
    )
    if 'token_semantic_loss_cls' in losses_base:
        assert torch.allclose(
            losses_base['token_semantic_loss_cls'],
            losses_base['token_semantic_loss_cls_main']
        )


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_mask_helper_supports_class_matching():
    torch.manual_seed(0)
    cache = TokenMatchingCache()
    helper = TokenMaskTaskHelper(
        mask_mode='semantic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='semantic_token_labels',
        class_label_offset=1,
        class_weight=4.0
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch, predictions_post = _build_batch_and_predictions()
    losses, _ = helper.training_step(
        batch,
        batch_idx=0,
        predictions_post=predictions_post
    )
    assert 'token_semantic_loss_mask_main' in losses
    matching = cache.get_stage(0, 'main')
    assert len(matching) == 1
    assert matching[0][0].numel() > 0


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_semantic_helper_uses_side_outputs():
    torch.manual_seed(0)
    cache = TokenMatchingCache()
    mask_helper = TokenMaskTaskHelper(
        mask_mode='semantic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='semantic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    semantic_helper = TokenSemanticTaskHelper(
        n_classes=2, matching_cache=cache
    )
    device = torch.device('cpu')
    mask_helper.initialize(device)
    semantic_helper.initialize(device)
    mask_helper.train()
    semantic_helper.train()
    batch, predictions_post = _build_batch_and_predictions()
    mask_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    cache.set_stage(0, 'aux_1', cache.get_stage(0, 'main'))
    predictions_with_aux = dict(predictions_post)
    predictions_with_aux['token_semantic_side_outputs'] = (
        predictions_post['token_semantic_output'],
    )
    losses_aux, _ = semantic_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_with_aux
    )
    losses_base, _ = semantic_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    total_key = semantic_helper.mark_as_total('token_semantic')
    assert 'token_semantic_loss_aux_1' in losses_aux
    aux_loss = losses_aux['token_semantic_loss_aux_1']
    main_loss = losses_aux['token_semantic_loss_main']
    semantic_coefficient = semantic_helper._semantic_coefficient
    expected_total = (aux_loss + main_loss) * semantic_coefficient
    assert torch.allclose(losses_aux[total_key], expected_total)
    assert torch.allclose(
        losses_base[total_key],
        losses_base['token_semantic_loss_main'] * semantic_coefficient
    )


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_semantic_validation_metrics():
    cache = TokenMatchingCache()
    mask_helper = TokenMaskTaskHelper(
        mask_mode='semantic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='semantic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    semantic_helper = TokenSemanticTaskHelper(
        n_classes=2,
        matching_cache=cache
    )
    device = torch.device('cpu')
    mask_helper.initialize(device)
    semantic_helper.initialize(device)
    mask_helper.train()
    semantic_helper.eval()
    batch, predictions_post = _build_batch_and_predictions()
    mask_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )

    semantic_helper.validation_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    artifacts, _, logs = semantic_helper.validation_epoch_end()
    assert 'token_semantic_miou' in logs
    assert 'token_semantic_ious_per_class' in artifacts


def test_token_panoptic_postprocessing_matches_targets():
    batch, predictions_post = _build_panoptic_batch_and_predictions()
    mask_post = postprocessing.TokenMaskPostprocessing()
    mask_outputs = mask_post.postprocess(
        (predictions_post['token_mask_output'], None),
        batch=batch,
        is_training=False
    )
    semantic_post = postprocessing.TokenSemanticPostprocessing()
    semantic_outputs = semantic_post.postprocess(
        (predictions_post['token_semantic_output'], ()),
        batch=batch,
        is_training=False,
        out_dict=mask_outputs
    )
    post = postprocessing.TokenPanopticPostprocessing(
        semantic_classes_is_thing=(True, True, True)
    )
    outputs = post.postprocess(
        data=(predictions_post['token_semantic_output'], ()),
        batch=batch,
        is_training=False,
        out_dict=semantic_outputs
    )
    seg = outputs['token_panoptic_segmentation'].cpu()
    # postprocessing groups the two queries into two distinct panoptic
    # segments matching the two-instance layout of the target panoptic
    assert seg.shape == batch['panoptic'].shape
    assert len(outputs['token_panoptic_id_dicts']) == seg.shape[0]
    # top row is one segment, bottom row another
    assert seg[0, 0, 0].item() == seg[0, 0, 1].item()
    assert seg[0, 1, 0].item() == seg[0, 1, 1].item()
    assert seg[0, 0, 0].item() != seg[0, 1, 0].item()
    id_dict = outputs['token_panoptic_id_dicts'][0]
    assert set(id_dict.keys()) == {
        seg[0, 0, 0].item(), seg[0, 1, 0].item()
    }


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_panoptic_helper_metrics():
    cache = TokenMatchingCache()
    mask_helper = TokenMaskTaskHelper(
        mask_mode='panoptic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='panoptic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    panoptic_helper = TokenPanopticTaskHelper(
        semantic_n_classes_without_void=3,
        semantic_classes_is_thing=(False, True, True, True),
        pq_n_workers=1,
        matching_cache=cache
    )
    device = torch.device('cpu')
    mask_helper.initialize(device)
    panoptic_helper.initialize(device)

    batch, predictions_post = _build_panoptic_batch_and_predictions()
    _attach_panoptic_predictions(
        predictions_post,
        semantic_classes_is_thing=(True, True, True)
    )
    mask_helper.train()
    panoptic_helper.train()
    mask_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    losses, _ = panoptic_helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_panoptic_loss_class' in losses
    assert 'token_panoptic_total_loss' in losses

    mask_helper.eval()
    panoptic_helper.eval()
    mask_helper.validation_step(
        batch, batch_idx=1, predictions_post=predictions_post
    )
    panoptic_helper.validation_step(
        batch, batch_idx=1, predictions_post=predictions_post
    )
    artifacts, _, logs = panoptic_helper.validation_epoch_end()
    assert 'token_panoptic_miou' in logs
    assert 'token_panoptic_ious_per_class' in artifacts


@pytest.mark.skipif(
    not IS_TRANSFORMERS_AVAILABLE,
    reason="transformers (HuggingFace) is not available"
)
def test_token_mask_helper_panoptic_mode_uses_panoptic_prefix():
    cache = TokenMatchingCache()
    helper = TokenMaskTaskHelper(
        mask_mode='panoptic',
        matching_cache=cache,
        class_scores_prefix='token_semantic',
        class_labels_key='panoptic_token_labels',
        class_label_offset=1,
        class_weight=0.0
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch, predictions_post = _build_batch_and_predictions()
    batch['panoptic_token_masks'] = [torch.ones(2, 2, 2)]
    batch['panoptic_token_labels'] = [torch.tensor([1, 1])]
    losses, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_panoptic_loss_mask' in losses
    assert 'token_panoptic_mask_total_loss' in losses


def test_token_orientation_helper_with_confidence():
    cache = TokenMatchingCache()
    indices = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    cache.set(0, [indices])
    helper = TokenOrientationTaskHelper(
        matching_cache=cache,
        supervise_confidence=True,
        vonmises_kappa=2.0
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch = {
        'panoptic_token_ids': [
            torch.tensor([1, 2], dtype=torch.long)
        ],
        'orientations': [
            {1: 0.0}
        ]
    }
    predictions_post = {
        'token_orientation_queries': torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32
        ),
        'token_orientation_confidence_logits': torch.zeros(1, 2)
    }
    losses, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_orientation_loss_direction' in losses
    assert 'token_orientation_loss_confidence' in losses
    total_key = helper.mark_as_total('token_orientation')
    assert total_key in losses
    helper.eval()
    helper.validation_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    _, _, logs = helper.validation_epoch_end()
    assert 'token_orientation_mae_rad' in logs
    assert 'token_orientation_mae_deg' in logs


def _build_visual_embedding_batch():
    # contiguous embedding LUT with one entry per instance
    lut = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32
    )
    # dense indices map each pixel to lut_idx + 1 (0 reserved for void)
    indices = torch.tensor(
        [[1, 1], [2, 2]],
        dtype=torch.int64
    )
    # one binary mask per instance, shaped [N, H, W]
    masks = torch.tensor([
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ], dtype=torch.float32)
    return {
        'dense_visual_embedding_lut': [lut],
        'dense_visual_embedding_indices': [indices],
        'panoptic_token_masks': [masks],
        'panoptic_token_labels': [torch.tensor([0, 1], dtype=torch.int64)],
    }


def test_token_visual_embedding_helper_losses():
    cache = TokenMatchingCache()
    indices = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    cache.set(0, [indices])
    helper = TokenEmbeddingTaskHelper(
        matching_cache=cache,
        n_classes=2,
        semantic_classes_is_thing=(False, True, True)
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch = _build_visual_embedding_batch()
    predictions_post = {
        'token_visual_embedding_output': torch.tensor(
            [[[1.0, 0.0], [0.5, 0.5]]],
            dtype=torch.float32
        ),
        'token_visual_embedding_side_outputs': ()
    }
    losses, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_visual_embedding_loss' in losses


def test_token_visual_embedding_helper_side_outputs():
    cache = TokenMatchingCache()
    indices = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    cache.set(0, [indices])
    # aux stages reuse the main-stage matching
    cache.set_stage(0, 'aux_1', cache.get_stage(0, 'main'))
    helper = TokenEmbeddingTaskHelper(
        matching_cache=cache,
        n_classes=2,
        semantic_classes_is_thing=(False, True, True)
    )
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch = _build_visual_embedding_batch()
    base_pred = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.5]]],
        dtype=torch.float32
    )
    predictions_post = {
        'token_visual_embedding_output': base_pred,
        'token_visual_embedding_side_outputs': (base_pred,)
    }
    losses_aux, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    predictions_post['token_visual_embedding_side_outputs'] = ()
    losses_base, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    total_key = helper.mark_as_total('token_visual_embedding')
    # side outputs contribute an aux loss stage
    assert 'token_visual_embedding_loss' in losses_aux
    assert 'token_visual_embedding_loss_aux_1' in losses_aux
    assert 'token_visual_embedding_loss_aux_1' not in losses_base
    # total loss is the sum of every stage loss component
    expected_aux = sum(
        v for k, v in losses_aux.items() if k != total_key
    )
    assert torch.allclose(losses_aux[total_key], expected_aux)
    expected_base = sum(
        v for k, v in losses_base.items() if k != total_key
    )
    assert torch.allclose(losses_base[total_key], expected_base)


def test_token_visual_embedding_helper_metrics():
    cache = TokenMatchingCache()
    indices = (torch.tensor([0]), torch.tensor([0]))
    cache.set(0, [indices])
    helper = TokenEmbeddingTaskHelper(
        matching_cache=cache,
        n_classes=2,
        semantic_classes_is_thing=(False, True, True)
    )
    helper.initialize(torch.device('cpu'))
    helper.eval()
    batch = {
        'dense_visual_embedding_lut': [
            torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ],
        'dense_visual_embedding_indices': [
            torch.tensor([[1]], dtype=torch.int64)
        ],
        'panoptic_token_masks': [
            torch.tensor([[[1]]], dtype=torch.float32)
        ],
        'panoptic_token_labels': [torch.tensor([0], dtype=torch.int64)],
        'semantic_fullres': torch.tensor([[[1]]], dtype=torch.long)
    }
    predictions_post = {
        'token_visual_embedding_output': torch.tensor(
            [[[1.0, 0.0]]],
            dtype=torch.float32
        ),
        'token_visual_embedding_side_outputs': (),
        get_fullres_key('token_visual_embedding_text_based_semantic_idx'):
            torch.tensor([[[0]]], dtype=torch.long),
        get_fullres_key(
            'token_visual_embedding_visual_mean_based_semantic_idx'
        ): torch.tensor([[[0]]], dtype=torch.long),
    }
    helper.validation_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    artifacts, _, logs = helper.validation_epoch_end()
    assert 'token_visual_embedding_text_based_miou' in logs
    assert 'token_visual_embedding_visual_mean_based_miou' in logs
    assert 'token_visual_embedding_text_based_ious_per_class' in artifacts
    assert (
        'token_visual_embedding_visual_mean_based_ious_per_class' in artifacts
    )


def test_token_scene_helper_losses():
    helper = TokenSceneTaskHelper(n_classes=3)
    helper.initialize(torch.device('cpu'))
    helper.train()
    batch = {
        'scene': torch.tensor([1, 2], dtype=torch.long)
    }
    logits = torch.randn(2, 3, requires_grad=True)
    predictions_post = {
        'scene_output': logits,
    }
    losses, _ = helper.training_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    assert 'token_scene_loss_main' in losses
    # TokenSceneDecoder does not provide side outputs, so no aux losses exist
    assert 'token_scene_loss_aux_1' not in losses
    total_key = helper.mark_as_total('token_scene')
    assert torch.allclose(losses[total_key], losses['token_scene_loss_main'])


def test_token_scene_helper_metrics():
    helper = TokenSceneTaskHelper(n_classes=2)
    helper.initialize(torch.device('cpu'))
    helper.eval()
    batch = {
        'scene': torch.tensor([1, 2], dtype=torch.long)
    }
    predictions_post = {
        'scene_output': torch.randn(2, 2),
        'scene_side_outputs': (),
        'scene_class_idx': torch.tensor([0, 1], dtype=torch.long)
    }
    helper.validation_step(
        batch, batch_idx=0, predictions_post=predictions_post
    )
    artifacts, _, logs = helper.validation_epoch_end()
    assert 'scene_acc' in logs
    assert 'scene_bacc' in logs
    assert 'scene_cm' in artifacts
