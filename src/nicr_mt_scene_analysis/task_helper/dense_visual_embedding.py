# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
"""
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..data.preprocessing.multiscale_supervision import get_downscale
from ..loss import L1Loss
from ..loss import MSELoss
from ..loss import CosineEmbeddingLoss
from ..metric import MeanIntersectionOverUnion
from ..types import BatchType
from .base import TaskHelperBase
from .base import append_detached_losses_to_logs
from .base import append_profile_to_logs
from ..visualization import visualize_semantic_pil
from ..visualization import visualize_heatmap_pil


KNOWN_DENSE_VISUAL_EMBEDDING_LOSS_FUNCTIONS = (
    'mse',
    'l1',
    'cos_emb',
)


class DenseVisualEmbeddingTaskHelper(TaskHelperBase):
    def __init__(
        self,
        n_classes: int,
        loss_name: str = 'cos_emb',
        disable_multiscale_supervision: bool = False,
        examples_cmap: Union[Sequence[Tuple[int, int, int]], np.array] = None
    ) -> None:
        super().__init__()
        self._loss_name = loss_name.lower()
        self._disable_multiscale_supervision = disable_multiscale_supervision

        # during validation, we store some examples for visualization purposes
        self._examples = {}
        self._n_classes = n_classes
        self._examples_cmap = examples_cmap

    def initialize(self, device: torch.device):
        assert self._loss_name in KNOWN_DENSE_VISUAL_EMBEDDING_LOSS_FUNCTIONS
        if self._loss_name == 'mse':
            self._loss = MSELoss()
        elif self._loss_name == 'l1':
            self._loss = L1Loss()
        elif self._loss_name == 'cos_emb':
            self._loss = CosineEmbeddingLoss()

        # metrics (keep it on cpu, it is faster)
        self._text_metric_iou = MeanIntersectionOverUnion(
            n_classes=self._n_classes
        )
        self._text_metric_iou.reset()

        self._visual_mean_metric_iou = MeanIntersectionOverUnion(
            n_classes=self._n_classes
        )
        self._visual_mean_metric_iou.reset()

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # collect predictions and targets for loss
        no_multiscale = self._disable_multiscale_supervision
        embedding_preds, keys, downscales = self.collect_predictions_for_loss(
            predictions_post=predictions_post,
            predictions_post_key='dense_visual_embedding_output',
            side_outputs_key=None if no_multiscale else 'dense_visual_embedding_side_outputs'
        )

        target_embedding_lut = self.collect_targets_for_loss(
            batch=batch,
            batch_key='dense_visual_embedding_lut',
            downscales=downscales
        )

        target_embedding_indices = [
            self._get_spatial_target_for_prediction(
                batch=batch,
                batch_key='dense_visual_embedding_indices',
                prediction=pred
            )
            for pred in embedding_preds
        ]

        # the embedding indices might contain 0 values which are
        # placeholders for void / invalid pixels. We do not want to
        # compute the loss for these pixels (as we don't have a target
        # embedding for them). Therefore, we create a mask and only
        # compute the loss for valid pixels.
        is_vaid_embedding = [x != 0 for x in target_embedding_indices]

        # mask out predictions to only contain pixels with targets as
        # we can't compue the loss if we dont have the target embedding.
        # permute is to convert from nchw to nhwc as we want to mask out
        # spatial but keep embedding vector.
        preds_masked = [
            x.permute(0, 2, 3, 1)[mask]
            for x, mask in zip(embedding_preds, is_vaid_embedding)
        ]

        # targets are stored as lut + indices, so we need to map
        # the indices to the lut to get the actual target embeddings.
        # we only keep the valid target embeddings.
        # We can only resolve the mapping by iterating over the scale and
        # batch as we need to combine it elment by element.
        # Note: The outer loop is iterating over the different scales.
        targets_masked = []
        for (
            scale_valid_embedding_mask,
            scale_target_embedding_indices,
            scale_target_embedding_lut
        ) in zip(
            is_vaid_embedding, target_embedding_indices, target_embedding_lut
        ):

            # shift indices by -1 as 0 was used as placeholder for invalid
            # (without gt) pixels, which we mask out. The shift makes access
            # to lut easier, as it starts with index 0 (shifted).
            keep_valid_indices = (scale_target_embedding_indices - 1)[scale_valid_embedding_mask]

            # get a list which element of the tensor belogs to which batch
            # which can be used for masking elements of a batch.
            # we just select index 0 as its the batch axis.
            batch_indices = torch.where(scale_valid_embedding_mask)[0]

            # lut is stored as a list of tensors (as they aren't stackable).
            # to create a joined target embedding we iterate over the batch.
            masked_target_vector = []
            for batch_idx in range(len(scale_target_embedding_lut)):
                # to save some vram (similar to index+lut) we want to
                # create the full target vector as late a possible.
                # because of that we create a mask for the current batch
                # element in the current scale.
                current_batch_element_mask = batch_indices == batch_idx

                # only take element with valid index
                current_batch_element_indices = keep_valid_indices[current_batch_element_mask]

                # There is a single SUN RGB-D sample which dosn't have
                # gt annotations. In this case there are no valid indices.
                # The sample just dons't contribute to the loss, by not getting
                # appended to keep_target_masked. This creates no alignment
                # issues as keep_target_masked is just a long vector.
                # We could instead just append an empty tensor, however
                # this would create issues with torch.cat.
                if len(current_batch_element_indices) == 0:
                    continue

                # finaly combine the index with the lut to get a target vector
                # and append it to the masked targets
                masked_target_vector.append(
                    scale_target_embedding_lut[batch_idx][current_batch_element_indices]
                )
            # make it to an actual torch tensor and append it to the list with
            # different scales.
            masked_target_vector = torch.cat(masked_target_vector, dim=0)
            targets_masked.append(masked_target_vector)

        # Compute losses on masked elements
        loss_outputs = self._loss(
            input_tensors=preds_masked, target_tensors=targets_masked
        )

        # Create loss dictionary
        loss_dict = {
            f'dense_visual_embedding_loss_{key}': loss/n
            for key, (loss, n) in zip(keys, loss_outputs)
        }

        # Compute total loss
        total_loss = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs],
            n_elements=[n for _, n in loss_outputs]
        )

        # Append unweighted total loss
        loss_dict[self.mark_as_total('dense_visual_embedding')] = total_loss

        return loss_dict

    def _get_spatial_target_for_prediction(
        self,
        batch: BatchType,
        batch_key: str,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        target_fullres = batch[batch_key]
        h_target, w_target = target_fullres.shape[-2:]
        h_pred, w_pred = prediction.shape[-2:]

        if h_pred == h_target and w_pred == w_target:
            return target_fullres

        assert (h_target % h_pred) == 0 and (w_target % w_pred) == 0, (
            "Prediction and target resolutions are incompatible: "
            f"{(h_pred, w_pred)} vs {(h_target, w_target)}"
        )

        downscale_h = h_target // h_pred
        downscale_w = w_target // w_pred

        assert downscale_h == downscale_w, (
            "Non-uniform scaling between height and width is not supported: "
            f"{downscale_h} vs {downscale_w}"
        )

        downscale_sample = get_downscale(batch, downscale_h)
        assert downscale_sample is not None and batch_key in downscale_sample, (
            f"Required downscale '{downscale_h}' for key '{batch_key}' "
            "is missing in batch. Ensure multiscale preprocessing is enabled."
        )
        return downscale_sample[batch_key]

    @append_profile_to_logs('semantic_embedding_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post
        )
        return loss_dict, {}

    @append_profile_to_logs('dense_visual_embedding_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post
        )

        # update metric
        target = get_fullres(batch, 'semantic')
        mask = target != 0    # mask of non-void pixels
        # Text-based semantic
        text_key = get_fullres_key('dense_visual_embedding_text_based_semantic_idx')
        if text_key in predictions_post:
            preds = predictions_post[text_key][mask]
            target_masked = target[mask] - 1    # first apply mask -> -1 is safe
            self._text_metric_iou.update(preds=preds.cpu(), target=target_masked.cpu())

        # Visual-mean-based semantic
        visual_key = get_fullres_key('dense_visual_embedding_visual_mean_based_semantic_idx')
        if visual_key in predictions_post:
            preds_visual = predictions_post[visual_key][mask]
            target_masked = target[mask] - 1
            self._visual_mean_metric_iou.update(
                preds=preds_visual.cpu(),
                target=target_masked.cpu()
            )

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            # For text based semantic segmentation
            if 'dense_visual_embedding_text_based_semantic_idx' in predictions_post:
                ex = predictions_post['dense_visual_embedding_text_based_semantic_idx'][0]
                key = f'dense_visual_embedding_text_based_example_batch_idx_{batch_idx}_0'
                self._examples[key] = visualize_semantic_pil(
                    semantic_img=ex.cpu().numpy(),
                    colors=self._examples_cmap
                )
            if 'dense_visual_embedding_text_based_semantic_score' in predictions_post:
                ex = predictions_post['dense_visual_embedding_text_based_semantic_score'][0]
                key = f'dense_visual_embedding_text_based_example_batch_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=ex.cpu().numpy(),
                    min_=0, max_=1
                )

            # For visual mean based semantic segmentation
            if 'dense_visual_embedding_visual_mean_based_semantic_idx' in predictions_post:
                ex = predictions_post['dense_visual_embedding_visual_mean_based_semantic_idx'][0]
                key = f'dense_visual_embedding_visual_mean_based_example_batch_idx_{batch_idx}_0'
                self._examples[key] = visualize_semantic_pil(
                    semantic_img=ex.cpu().numpy(),
                    colors=self._examples_cmap
                )
            if 'dense_visual_embedding_visual_mean_based_semantic_score' in predictions_post:
                ex = predictions_post['dense_visual_embedding_visual_mean_based_semantic_score'][0]
                key = f'dense_visual_embedding_visual_mean_based_example_batch_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=ex.cpu().numpy(),
                    min_=0, max_=1
                )

        return loss_dict, {}

    @append_profile_to_logs('semantic_epoch_end_time')
    def validation_epoch_end(self):
        miou, ious = self._text_metric_iou.compute(return_ious=True)
        visual_miou, visual_ious = self._visual_mean_metric_iou.compute(return_ious=True)
        logs = {
            'dense_visual_embedding_text_based_miou': miou,
            'dense_visual_embedding_visual_mean_based_miou': visual_miou
        }
        artifacts = {
            # For text based results
            'dense_visual_embedding_text_based_semantic_cm': self._text_metric_iou.confmat.clone(),
            'dense_visual_embedding_text_based_semantic_ious_per_class': ious.clone(),
            # For visual mean based results
            'dense_visual_embedding_visual_mean_based_semantic_cm': self._visual_mean_metric_iou.confmat.clone(),
            'dense_visual_embedding_visual_mean_based_semantic_ious_per_class': visual_ious.clone(),
        }

        # reset metric (it is not done automatically)
        self._text_metric_iou.reset()
        assert self._text_metric_iou.confmat.sum() == 0

        self._visual_mean_metric_iou.reset()
        assert self._visual_mean_metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs
