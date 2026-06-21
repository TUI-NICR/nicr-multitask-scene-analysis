# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import numpy as np

from nicr_mt_scene_analysis.data.preprocessing.token import (
    PanopticTokenMaskTargetGenerator
)
from nicr_mt_scene_analysis.data.preprocessing.token import (
    SemanticTokenMaskTargetGenerator
)


def test_semantic_token_mask_generator_builds_masks_and_labels():
    semantic = np.array([
        [0, 1, 1],
        [2, 2, 0],
    ], dtype=np.int32)
    generator = SemanticTokenMaskTargetGenerator()
    sample = {'semantic': semantic.copy()}
    result = generator(sample)
    labels = result['semantic_token_labels']
    np.testing.assert_array_equal(labels, np.array([1, 2], dtype=np.int64))
    masks = result['semantic_token_masks']
    assert masks.shape == (2, 3, 2)
    np.testing.assert_array_equal(
        masks[..., 0],
        np.array([
            [0., 1., 1.],
            [0., 0., 0.],
        ], dtype=np.float32)
    )


def test_semantic_token_mask_generator_respects_ignore_labels():
    semantic = np.array([
        [5, 5],
        [0, 0],
    ])
    generator = SemanticTokenMaskTargetGenerator(ignore_labels=(0, 5))
    sample = {'semantic': semantic.copy()}
    result = generator(sample)
    labels = result['semantic_token_labels']
    masks = result['semantic_token_masks']
    assert labels.shape == (0,)
    assert masks.shape == (2, 2, 0)


def test_panoptic_token_mask_generator_extracts_masks():
    max_instances = 1 << 8
    panoptic = np.array([
        [1 * max_instances + 1, 1 * max_instances + 1],
        [2 * max_instances + 5, 0]
    ], dtype=np.int64)
    generator = PanopticTokenMaskTargetGenerator(
        max_instances_per_category=max_instances
    )
    sample = {'panoptic': panoptic.copy()}
    result = generator(sample)
    labels = result['panoptic_token_labels']
    np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int64))
    ids = result['panoptic_token_ids']
    np.testing.assert_array_equal(
        ids,
        np.array([
            1 * max_instances + 1,
            2 * max_instances + 5
        ], dtype=np.int64)
    )
    masks = result['panoptic_token_masks']
    assert masks.shape == (2, 2, 2)
