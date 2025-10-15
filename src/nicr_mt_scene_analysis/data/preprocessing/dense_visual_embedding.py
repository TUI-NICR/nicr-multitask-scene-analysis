# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np

from .base import PreprocessingBase
from .utils import _keys_available
from ...types import BatchType


class DenseVisualEmbeddingTargetGenerator(PreprocessingBase):
    def __init__(
        self,
        diff_factor: float = 0.65,
        multiscale_processing: bool = True,
    ):
        super().__init__(multiscale_processing=multiscale_processing)
        self.diff_factor = diff_factor

    def _process_scale(
        self,
        panoptic_embedding_targets: Dict[int, np.ndarray],
        panoptic_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Store embedding vectors in a single numpy array which
        # can be used as contiguous lookup table (LUT).
        embeddings = np.array(
            list(panoptic_embedding_targets.values()),
            dtype=np.float32
        )

        # Map the panoptic target to the indices of the contiguous LUT.
        # Basically it just maps relatively sparse panoptic ids to dense
        # indices.
        keys = [int(key) for key in panoptic_embedding_targets.keys()]
        indices = np.full(panoptic_target.shape, 0, dtype=np.int32)
        for idx, key in enumerate(keys):
            mask = panoptic_target == key
            if not np.any(mask):
                continue
            indices[mask] = idx + 1  # +1 to reserve 0 for background/void

        return embeddings, indices

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        # Already normalized image embedding
        if not _keys_available(sample, ('image_embedding',)):
            # might be an inference call
            return sample, {}
        image_embedding = sample['image_embedding']
        # Get the panoptic embeddings
        if not _keys_available(sample, ('panoptic_embedding',)):
            # might be an inference call
            return sample, {}
        panoptic_embeddings_target = sample['panoptic_embedding']

        # Adjust panoptic embeddings by subtracting the image embedding as
        # described in the paper. This helps to reduce global scene context
        # and helps to focus on local features.
        panoptic_embeddings_target = {
            k: v - self.diff_factor * image_embedding
            for k, v in panoptic_embeddings_target.items()
        }

        # Normalize panoptic embeddings
        panoptic_embeddings_target = {
            k: v / np.linalg.norm(v, axis=-1, keepdims=True)
            for k, v in panoptic_embeddings_target.items()
        }

        # Generate dense embeddings
        panoptic_target = sample['panoptic']

        embedding_lut, embedding_indices = self._process_scale(
            panoptic_embeddings_target, panoptic_target
        )

        # Due to high memory consumption, we do not store the a dense
        # target, but only the lookup table and indices.
        # The dense target can be reconstructed from the LUT and indices.
        sample['dense_visual_embedding_lut'] = embedding_lut
        sample['dense_visual_embedding_indices'] = embedding_indices

        return sample, {}
