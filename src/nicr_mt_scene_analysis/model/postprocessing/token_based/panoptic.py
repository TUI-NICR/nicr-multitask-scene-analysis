# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Sequence

import torch

from ....data.preprocessing.resize import get_fullres_key
from ....data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from .semantic import TokenSemanticPostprocessing


class TokenPanopticPostprocessing(TokenSemanticPostprocessing):
    def __init__(
        self,
        *,
        n_classes: Optional[int] = None,
        semantic_classes_is_thing: Sequence[bool],
        mask_threshold: float = 0.8,
        mask_pixel_threshold: float = 0.1,
        overlap_threshold: float = 0.8,
        max_instances_per_category: int = 1 << 16
    ):
        super().__init__(n_classes=n_classes)
        self._classes_is_thing = tuple(
            bool(v) for v in semantic_classes_is_thing
        )
        self._mask_threshold = mask_threshold
        self._mask_pixel_threshold = mask_pixel_threshold
        self._overlap_threshold = overlap_threshold
        self._max_instances_per_category = max_instances_per_category

    @property
    def max_instances_per_category(self):
        return self._max_instances_per_category

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        # reuse the semantic postprocessing to fill the shared out_dict and
        # then assemble the panoptic segmentation from query masks + class
        # scores.
        out_dict = super()._postprocess_inference(
            data, batch, out_dict=out_dict
        )
        mask_probs = out_dict['token_mask_probs']
        class_scores = out_dict.get('token_semantic_softmax_scores')
        if class_scores is None:
            logits = out_dict.get('token_semantic_output')
            assert logits is not None, (
                'Token panoptic postprocessing expects semantic logits or '
                'softmax scores in the shared postprocessing context.'
            )
            class_scores = torch.softmax(logits, dim=-1)
        assert mask_probs.ndim == 4, 'mask_probs must be shaped [B, Q, H, W].'
        assert class_scores.ndim == 3, 'class_scores must be shaped [B, Q, C].'
        assert mask_probs.shape[:2] == class_scores.shape[:2], (
            'mask_probs and class_scores batch/query dims must match.'
        )

        # top-1 class per query
        scores, labels = torch.max(class_scores, dim=-1)
        batch_panoptic = []
        batch_ids = []
        batch_scores = []
        batch_queries = []
        thing_flags = tuple(bool(v) for v in self._classes_is_thing)
        for masks, cls_scores, cls_labels in zip(mask_probs, scores, labels):
            # drop queries below the class-score threshold
            keep = cls_scores > self._mask_threshold
            h, w = masks.shape[-2:]
            panoptic = torch.zeros(
                (h, w),
                device=masks.device,
                dtype=torch.long
            )
            ids = {}
            segment_scores = {}
            segment_queries = {}
            if not keep.any():
                batch_panoptic.append(panoptic.cpu())
                batch_ids.append(ids)
                batch_scores.append(segment_scores)
                batch_queries.append(segment_queries)
                continue
            # assign each pixel to the surviving query with the highest
            # class-weighted mask probability
            keep_idx = keep.nonzero(as_tuple=False).flatten()
            mask_ids = (
                cls_scores[keep_idx][..., None, None] * masks[keep_idx]
            ).argmax(0)
            segments = torch.full(
                (h, w),
                fill_value=-1,
                device=masks.device,
                dtype=torch.long
            )
            # build per-segment masks. discard segments that lose too much
            # area to the per-pixel assignment, and merge all queries that
            # map to the same stuff class into one segment.
            stuff_segment_ids = {}
            segment_id = 0
            segment_and_class_ids = []
            for k, class_id in enumerate(cls_labels[keep_idx].tolist()):
                orig_mask = masks[keep_idx][k] >= self._mask_pixel_threshold
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask
                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < self._overlap_threshold
                ):
                    continue
                if not thing_flags[class_id]:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    stuff_segment_ids[class_id] = segment_id
                segments[final_mask] = segment_id
                query_idx = int(keep_idx[k].detach().cpu())
                segment_and_class_ids.append(
                    (segment_id, class_id, k, query_idx)
                )
                segment_id += 1

            # encode each segment as class_id * max_instances + instance_id
            for (
                seg_id, class_id, keep_local_idx, query_idx
            ) in segment_and_class_ids:
                segment_mask = segments == seg_id
                panoptic_class_id = class_id + 1
                pan_id = (
                    panoptic_class_id
                    * self._max_instances_per_category
                    + seg_id
                )
                pan_id_tensor = torch.tensor(
                    pan_id,
                    device=panoptic.device,
                    dtype=panoptic.dtype
                )
                panoptic = torch.where(
                    segment_mask,
                    pan_id_tensor,
                    panoptic
                )
                ids[int(pan_id)] = int(seg_id)
                segment_scores[int(pan_id)] = float(
                    cls_scores[keep_idx][keep_local_idx].detach().cpu()
                )
                segment_queries[int(pan_id)] = query_idx
            batch_panoptic.append(panoptic.cpu())
            batch_ids.append(ids)
            batch_scores.append(segment_scores)
            batch_queries.append(segment_queries)
        panoptic_tensor = torch.stack(batch_panoptic)
        out_dict.update({
            'token_panoptic_segmentation': panoptic_tensor,
            'token_panoptic_id_dicts': batch_ids,
            'token_panoptic_score_dicts': batch_scores,
            'token_panoptic_query_dicts': batch_queries
        })
        # resize from the network's working resolution back to the original
        # image size for the evaluation metrics
        crop_slices, resize_shape = get_valid_region_slices_and_fullres_shape(
            batch, 'panoptic'
        )
        panoptic_fullres = self._crop_to_valid_region_and_resize_prediction(
            panoptic_tensor,
            valid_region_slices=crop_slices,
            shape=resize_shape,
            mode='nearest'
        )
        fullres_key = get_fullres_key('token_panoptic_segmentation')
        out_dict[fullres_key] = panoptic_fullres
        return out_dict
