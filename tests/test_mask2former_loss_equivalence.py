# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from types import SimpleNamespace

import pytest
import torch

from nicr_mt_scene_analysis.task_helper import TokenSemanticTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenMaskTaskHelper
from nicr_mt_scene_analysis.task_helper.token_based import TokenMatchingCache

_transformers = pytest.importorskip("transformers")
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss


def _make_random_masks(
    *,
    n_masks: int,
    height: int,
    width: int,
    device: torch.device
) -> torch.Tensor:
    masks = (torch.rand(n_masks, height, width, device=device) > 0.7)
    if n_masks == 0:
        return masks.float()
    # ensure each mask has at least one pixel
    for idx in range(n_masks):
        if not masks[idx].any():
            y = torch.randint(0, height, (1,))
            x = torch.randint(0, width, (1,))
            masks[idx, y, x] = True
    return masks.float()


def _reference_matched_cross_entropy(
    *,
    class_queries_logits: torch.Tensor,
    class_labels,
    matching
) -> torch.Tensor:
    # mean cross-entropy over matched queries. Softmax spans all n_classes+1
    # logits because the helper does not slice off the trailing no-object logit.
    log_probs = torch.log_softmax(class_queries_logits, dim=-1)
    picked = []
    for sample_idx, (src_indices, tgt_indices) in enumerate(matching):
        if src_indices.numel() == 0:
            continue
        labels = class_labels[sample_idx][tgt_indices]
        for query_idx, label in zip(src_indices.tolist(), labels.tolist()):
            picked.append(-log_probs[sample_idx, query_idx, label])
    assert picked, "expected at least one matched query in the reference"
    return torch.stack(picked).mean()


def test_mask2former_loss_matches_helpers():
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size = 2
    n_queries = 5
    n_classes = 3
    height = 8
    width = 8
    n_points = 16
    mask_weight = 5.0
    dice_weight = 5.0
    class_weight = 4.0
    no_object_weight = 0.1

    masks_queries_logits = torch.randn(
        batch_size, n_queries, height, width, device=device
    )
    class_queries_logits = torch.randn(
        batch_size, n_queries, n_classes + 1, device=device
    )

    mask_labels = []
    class_labels = []
    batch_masks = []
    batch_labels = []
    for _ in range(batch_size):
        n_targets = int(torch.randint(1, 4, (1,)).item())
        masks = _make_random_masks(
            n_masks=n_targets,
            height=height,
            width=width,
            device=device
        )
        labels = torch.randint(0, n_classes, (n_targets,), device=device)
        mask_labels.append(masks)
        class_labels.append(labels)
        batch_masks.append(masks)
        # batch labels are 1-indexed (0 is void). The helpers strip the offset.
        batch_labels.append(labels + 1)

    config = SimpleNamespace(
        num_labels=n_classes,
        no_object_weight=no_object_weight,
        train_num_points=n_points,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        class_weight=class_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight
    )
    weight_dict = {
        "loss_mask": mask_weight,
        "loss_dice": dice_weight,
        "loss_cross_entropy": class_weight,
    }

    torch.manual_seed(0)
    reference_loss = Mask2FormerLoss(config, weight_dict)
    losses_ref = reference_loss(
        masks_queries_logits,
        class_queries_logits,
        mask_labels,
        class_labels
    )

    matching_cache = TokenMatchingCache()
    mask_helper = TokenMaskTaskHelper(
        mask_mode="semantic",
        matching_cache=matching_cache,
        n_points=n_points,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=mask_weight,
        dice_coefficient=dice_weight,
        class_scores_prefix="token_semantic",
        class_labels_key="semantic_token_labels",
        class_label_offset=1,
        class_weight=class_weight
    )
    semantic_helper = TokenSemanticTaskHelper(
        n_classes=n_classes,
        matching_cache=matching_cache,
        semantic_coefficient=class_weight
    )
    mask_helper.initialize(device)
    semantic_helper.initialize(device)

    batch = {
        "semantic_token_masks": batch_masks,
        "semantic_token_labels": batch_labels,
    }
    predictions_post = {
        "token_mask_output": masks_queries_logits,
        "token_mask_side_outputs": tuple(),
        "token_semantic_output": class_queries_logits,
        "token_semantic_side_outputs": tuple(),
    }

    # re-seed so the helper's uncertainty point sampling matches the HF
    # reference
    torch.manual_seed(0)
    mask_losses = mask_helper.compute_mask_losses(batch, 0, predictions_post)
    semantic_losses = semantic_helper._compute_losses(
        batch,
        0,
        predictions_post
    )

    assert torch.allclose(
        mask_losses["token_semantic_loss_mask"],
        mask_weight * losses_ref["loss_mask"],
        rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        mask_losses["token_semantic_loss_dice"],
        dice_weight * losses_ref["loss_dice"],
        rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        mask_losses["token_semantic_mask_total_loss"],
        mask_weight * losses_ref["loss_mask"]
        + dice_weight * losses_ref["loss_dice"],
        rtol=1e-5, atol=1e-6
    )

    matching = matching_cache.get_stage(0, "main")
    expected_ce = _reference_matched_cross_entropy(
        class_queries_logits=class_queries_logits,
        class_labels=class_labels,
        matching=matching
    )
    expected_semantic = class_weight * expected_ce
    assert torch.allclose(
        semantic_losses["token_semantic_loss_main"],
        expected_ce,
        rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        semantic_losses["token_semantic_total_loss"],
        expected_semantic,
        rtol=1e-5, atol=1e-6
    )

    total_helper = (
        mask_losses["token_semantic_mask_total_loss"]
        + semantic_losses["token_semantic_total_loss"]
    )
    expected_total = (
        mask_weight * losses_ref["loss_mask"]
        + dice_weight * losses_ref["loss_dice"]
        + expected_semantic
    )
    assert torch.allclose(total_helper, expected_total, rtol=1e-5, atol=1e-6)
