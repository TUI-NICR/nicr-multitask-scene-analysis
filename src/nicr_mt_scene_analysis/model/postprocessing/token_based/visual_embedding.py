# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ....types import BatchType
from ....types import DecoderRawOutputType
from ....types import PostprocessingOutputType
from ....data.preprocessing.resize import get_fullres_key
from ....data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from ..dense_base import DensePostprocessingBase
from .base import TokenPostprocessingBase
from .semantic import add_dense_semantic_outputs


class TokenVisualEmbeddingPostprocessing(
    TokenPostprocessingBase,
    DensePostprocessingBase
):

    def __init__(
        self,
        *,
        text_embeddings_per_class: Optional[torch.Tensor] = None,
        mean_visual_embedding_per_class: Optional[torch.Tensor] = None,
        semantic_classes_is_thing: Optional[Sequence[bool]] = None,
        mask_threshold: float = 0.8,
        mask_pixel_threshold: float = 0.1,
        overlap_threshold: float = 0.8,
        query_mask_score_exponent: float = 0.75,
        similarity_score_scale: float = 20.0,
        max_instances_per_category: int = 1 << 16,
        linear_probing_head: Optional[nn.Module] = None
    ):
        super().__init__(output_key='token_visual_embedding_output')
        self._text_embeddings = text_embeddings_per_class
        self._mean_visual_embeddings = mean_visual_embedding_per_class
        self._classes_is_thing = None
        if semantic_classes_is_thing is not None:
            self._classes_is_thing = tuple(
                bool(v) for v in semantic_classes_is_thing
            )
        self._mask_threshold = mask_threshold
        self._mask_pixel_threshold = mask_pixel_threshold
        self._overlap_threshold = overlap_threshold
        self._query_mask_score_exponent = query_mask_score_exponent
        self._similarity_score_scale = similarity_score_scale
        self._max_instances_per_category = max_instances_per_category
        self._linear_probing_head = linear_probing_head

    @property
    def max_instances_per_category(self):
        return self._max_instances_per_category

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        # forward the raw query embedding (and the linear-probing logits if
        # available) plus a cosine-similarity matcher input that the loss can
        # use to assign queries to classes
        output, side_outputs = data
        linear_logits = None
        if isinstance(output, tuple):
            if len(output) >= 2:
                output, linear_logits = output[:2]
        if linear_logits is None and self._linear_probing_head is not None:
            linear_logits = self._linear_probing_head(output.detach())
        out_dict.update({
            'token_visual_embedding_output': output,
        })
        out_dict['token_visual_embedding_side_outputs'] = tuple(
            side_outputs or ()
        )
        if linear_logits is not None:
            out_dict['token_visual_embedding_linear_probing_logits'] = (
                linear_logits
            )
        if (
            'token_mask_output' not in out_dict
            and 'token_mask_probs' not in out_dict
        ):
            return out_dict
        ref_embeddings = self._mean_visual_embeddings
        if ref_embeddings is not None:
            ref_embeddings = ref_embeddings.to(output.device)
            ref_embeddings = ref_embeddings / ref_embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            output_norm = output / output.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            matcher_scores = torch.einsum(
                'bqd,cd->bqc',
                output_norm,
                ref_embeddings
            )
            matcher_side = tuple(
                torch.einsum(
                    'bqd,cd->bqc',
                    side / side.norm(dim=-1, keepdim=True).clamp_min(1e-6),
                    ref_embeddings
                )
                for side in tuple(side_outputs or ())
            )
            out_dict['token_visual_embedding_matcher_output'] = matcher_scores
            out_dict[
                'token_visual_embedding_matcher_side_outputs'
            ] = matcher_side
        return out_dict

    def _add_semantic_outputs(
        self,
        *,
        class_logits: torch.Tensor,
        mask_probs: torch.Tensor,
        batch: BatchType,
        prefix: str,
        out_dict: PostprocessingOutputType
    ) -> None:
        class_scores = torch.softmax(class_logits, dim=-1)
        dense_scores = torch.einsum('bqhw,bqc->bchw', mask_probs, class_scores)
        dense_scores_max, dense_labels = torch.max(dense_scores, dim=1)
        add_dense_semantic_outputs(
            dense_postprocessor=self,
            prefix=prefix,
            dense_scores=dense_scores,
            dense_score_max=dense_scores_max,
            dense_labels=dense_labels,
            batch=batch,
            out_dict=out_dict,
            fullres_mode='nearest'
        )

    def _add_semantic_outputs_from_labels(
        self,
        *,
        class_labels: torch.Tensor,
        n_classes: int,
        mask_probs: torch.Tensor,
        batch: BatchType,
        prefix: str,
        out_dict: PostprocessingOutputType
    ) -> None:
        # assign each pixel to the max mask and map that mask to its class.
        mask_ids = mask_probs.argmax(dim=1)
        dense_labels = torch.gather(
            class_labels,
            dim=1,
            index=mask_ids.view(mask_ids.shape[0], -1),
        ).view_as(mask_ids)
        dense_scores = torch.nn.functional.one_hot(
            dense_labels, num_classes=n_classes
        ).permute(0, 3, 1, 2).to(mask_probs.dtype)
        dense_scores_max = dense_scores.max(dim=1).values
        add_dense_semantic_outputs(
            dense_postprocessor=self,
            prefix=prefix,
            dense_scores=dense_scores,
            dense_score_max=dense_scores_max,
            dense_labels=dense_labels,
            batch=batch,
            out_dict=out_dict,
            fullres_mode='bilinear'
        )

    def _add_panoptic_outputs(
        self,
        *,
        class_scores: torch.Tensor,
        mask_probs: torch.Tensor,
        batch: BatchType,
        prefix: str,
        out_dict: PostprocessingOutputType,
        query_scores: Optional[torch.Tensor] = None
    ) -> None:
        # same panoptic assembly logic as TokenPanopticPostprocessing, but
        # using the per-query class scores derived from one of the embedding
        # reference sets instead of a learned classifier head.
        if self._classes_is_thing is None:
            return
        scores, labels = torch.max(class_scores, dim=-1)
        if query_scores is not None:
            scores = query_scores
        batch_panoptic = []
        batch_ids = []
        batch_scores = []
        batch_queries = []
        thing_flags = self._classes_is_thing
        for masks, cls_scores, cls_labels in zip(mask_probs, scores, labels):
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
            keep_idx = keep.nonzero(as_tuple=False).flatten()
            competition_masks = masks[keep_idx] * (
                masks[keep_idx] >= self._mask_pixel_threshold
            )
            mask_ids = (
                cls_scores[keep_idx][..., None, None] * competition_masks
            ).argmax(0)
            segments = torch.full(
                (h, w),
                fill_value=-1,
                device=masks.device,
                dtype=torch.long
            )
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
        out_dict.update({
            f'{prefix}_panoptic_segmentation': torch.stack(batch_panoptic),
            f'{prefix}_panoptic_id_dicts': batch_ids,
            f'{prefix}_panoptic_score_dicts': batch_scores,
            f'{prefix}_panoptic_query_dicts': batch_queries,
        })
        crop_slices, resize_shape = get_valid_region_slices_and_fullres_shape(
            batch, 'panoptic'
        )
        panoptic_fullres = self._crop_to_valid_region_and_resize_prediction(
            out_dict[f'{prefix}_panoptic_segmentation'],
            valid_region_slices=crop_slices,
            shape=resize_shape,
            mode='nearest'
        )
        fullres_key = get_fullres_key(f'{prefix}_panoptic_segmentation')
        out_dict[fullres_key] = panoptic_fullres

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        # for each available class-reference set (text / visual-mean /
        # linear-probing) produce one semantic and one panoptic prediction
        # under a `token_visual_embedding_<source>_*` prefix.
        out_dict = self._postprocess_training(
            data, batch, out_dict=out_dict
        )
        output = out_dict['token_visual_embedding_output']
        mask_probs = out_dict.get('token_mask_probs')
        if mask_probs is None:
            return out_dict
        mask_scores = mask_probs.mean(dim=(-2, -1))
        query_mask_scores = mask_scores.clamp_min(0).pow(
            self._query_mask_score_exponent
        )
        output = output / output.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if self._text_embeddings is not None:
            text_embeddings = self._text_embeddings.to(output.device)
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            text_similarities = torch.einsum(
                'bqd,cd->bqc',
                output,
                text_embeddings
            )
            text_panoptic_scores = torch.softmax(
                text_similarities * self._similarity_score_scale,
                dim=-1
            )
            text_labels = text_similarities.argmax(dim=-1)
            self._add_semantic_outputs_from_labels(
                class_labels=text_labels,
                n_classes=text_similarities.shape[-1],
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_text_based_semantic',
                out_dict=out_dict
            )
            self._add_panoptic_outputs(
                class_scores=text_panoptic_scores,
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_text_based',
                out_dict=out_dict,
                query_scores=(
                    text_panoptic_scores.max(dim=-1).values
                    * query_mask_scores
                )
            )
        if self._mean_visual_embeddings is not None:
            mean_embeddings = self._mean_visual_embeddings.to(output.device)
            mean_embeddings = mean_embeddings / mean_embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            visual_similarities = torch.einsum(
                'bqd,cd->bqc',
                output,
                mean_embeddings
            )
            visual_panoptic_scores = torch.softmax(
                visual_similarities * self._similarity_score_scale,
                dim=-1
            )
            visual_labels = visual_similarities.argmax(dim=-1)
            self._add_semantic_outputs_from_labels(
                class_labels=visual_labels,
                n_classes=visual_similarities.shape[-1],
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_visual_mean_based_semantic',
                out_dict=out_dict
            )
            self._add_panoptic_outputs(
                class_scores=visual_panoptic_scores,
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_visual_mean_based',
                out_dict=out_dict,
                query_scores=(
                    visual_panoptic_scores.max(dim=-1).values
                    * query_mask_scores
                )
            )
        linear_logits = out_dict.get(
            'token_visual_embedding_linear_probing_logits'
        )
        if linear_logits is not None:
            linear_scores = torch.softmax(linear_logits, dim=-1)
            self._add_semantic_outputs(
                class_logits=linear_logits,
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_linear_probing_semantic',
                out_dict=out_dict
            )
            self._add_panoptic_outputs(
                class_scores=linear_scores,
                mask_probs=mask_probs,
                batch=batch,
                prefix='token_visual_embedding_linear_probing',
                out_dict=out_dict
            )
        return out_dict



class TokenImageEmbeddingPostprocessing(TokenPostprocessingBase):
    def __init__(
        self,
        mean_image_embeddings_per_scene_class: Optional[torch.Tensor] = None,
        scene_text_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__(output_key='token_image_embedding_output')

        self.mean_image_embeddings_per_scene_class = (
            mean_image_embeddings_per_scene_class
        )
        self.scene_text_embeddings = scene_text_embeddings

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        (image_embeddings, scene_logits), _ = data
        out_dict['token_image_embedding_output'] = image_embeddings
        out_dict['token_image_embedding_linear_probing_logits'] = scene_logits
        out_dict['token_image_embedding_side_outputs'] = ()

        return out_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        *,
        out_dict: PostprocessingOutputType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data
        image_embeddings, scene_logits = output

        # create results dict
        out_dict.update({
            'token_image_embedding_output': image_embeddings,
            'token_image_embedding_linear_probing_logits': scene_logits,
            'token_image_embedding_side_outputs': side_outputs
        })

        if scene_logits is not None:
            scores_and_idx = [torch.max(sl, dim=-1) for sl in scene_logits]
            scores = torch.stack([s for s, _ in scores_and_idx], dim=1)
            idx = torch.stack([i for _, i in scores_and_idx], dim=1)
            out_dict.update({
                'token_image_embedding_linear_probing_score': scores,
                'token_image_embedding_linear_probing_idx': idx,
            })

        # normalize output anlong channel dim
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

        # add extra outputs matching estimated image embeddings to scene classes
        # through text embeddings and mean image embeddings per class
        if self.scene_text_embeddings is not None:
            text_embeddings = self.scene_text_embeddings

            # compute similarity through dot product
            scene_logits = torch.einsum(
                'bad,cd->bc',
                image_embeddings,
                text_embeddings
            )    # (batch_size, n_scene_classes)
            sim, idx = torch.max(scene_logits, dim=1)

            out_dict.update({
                'token_image_embedding_text_based_scene_sim': sim,
                'token_image_embedding_text_based_scene_idx': idx,
            })

        if self.mean_image_embeddings_per_scene_class is not None:
            mean_image_embeddings = self.mean_image_embeddings_per_scene_class

            # compute similarity through dot product
            scene_logits = torch.einsum(
                'bad,cd->bc',
                image_embeddings,
                mean_image_embeddings
            )    # (batch_size, n_scene_classes)
            sim, idx = torch.max(scene_logits, dim=1)

            out_dict.update({
                'token_image_embedding_visual_mean_based_scene_sim': sim,
                'token_image_embedding_visual_mean_based_scene_idx': idx,
            })

        return out_dict
