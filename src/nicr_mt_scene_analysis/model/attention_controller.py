# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Callable, Dict, Optional, Sequence, Tuple

import abc

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class AttentionMaskController(nn.Module, abc.ABC):
    def __init__(
        self,
        *,
        modalities: Sequence[str],
        start_stage_idx: int,
        end_stage_idx: int,
        stage_schedule: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        super().__init__()
        self._modalities = tuple(modalities)
        self._start_stage_idx = start_stage_idx
        self._end_stage_idx = end_stage_idx
        self._stage_schedule = stage_schedule or {}
        # Fraction of completed optimizer steps over the full training
        # schedule, with 0 at the first step and 1 at the last one. The
        # per-stage mask schedule uses this value to anneal masking. Store it
        # as a 0-d buffer instead of a Python float so torch.compile does not
        # recompile per progress value.
        self.register_buffer(
            '_progress',
            torch.zeros((), dtype=torch.float32),
            persistent=False
        )

    def should_apply(self, modality: str, stage_idx: int) -> bool:
        return (
            modality in self._modalities
            and self._start_stage_idx <= stage_idx <= self._end_stage_idx
        )

    def update_progress(self, progress: float) -> None:
        # Intended to be called by the training loop with
        # current_step / (total_steps - 1). Inference scripts set the same
        # schedule position from the checkpoint.
        self._progress.fill_(float(max(0.0, min(1.0, progress))))

    def _probability_for_stage(self, stage_idx: int) -> Tensor:
        # 0-d tensor return so callers stay branch-free under torch.compile.
        device = self._progress.device
        # Each stage has its own progress window. Before the window, all
        # queries keep mask-restricted attention. After the window, all queries
        # use unrestricted attention.
        window = self._stage_schedule[stage_idx]
        if window is None:
            # None disables scheduled masking for this stage.
            return torch.zeros((), device=device)
        start_ratio, end_ratio = window
        span = end_ratio - start_ratio
        if span <= 0:
            return torch.zeros((), device=device)
        # During the window, reduce the probability that a query keeps
        # mask-restricted attention from 1 to 0.
        rel = (self._progress - start_ratio) / span
        return (1.0 - rel).clamp(0.0, 1.0)

    @abc.abstractmethod
    def build_mask(
        self,
        *,
        queries: Tensor,
        patch_tokens: Tensor,
        grid_size: Tuple[int, int],
        n_prefix_tokens: int,
        modality: str,
        stage_idx: int
    ) -> Optional[Tensor]:
        # return an attention mask of shape [B, 1, N, N] or None.
        pass

class EOMTAttentionMaskController(AttentionMaskController):
    # masked attention following EoMT (Kerssies et al., CVPR 2025): the
    # decoder mask head gates which patch tokens each query may attend to, and
    # the masking probability is annealed over the injected stages.

    def __init__(
        self,
        *,
        modality: str,
        mask_logits_fn: Callable[[Tensor, Tensor, Tuple[int, int]], Tensor],
        start_stage_idx: int,
        end_stage_idx: int,
        stage_schedule: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        super().__init__(
            modalities=(modality,),
            start_stage_idx=start_stage_idx,
            end_stage_idx=end_stage_idx,
            stage_schedule=stage_schedule
        )
        self._mask_logits_fn = mask_logits_fn

    def build_mask(
        self,
        *,
        queries: Tensor,
        patch_tokens: Tensor,
        grid_size: Tuple[int, int],
        n_prefix_tokens: int,
        modality: str,
        stage_idx: int
    ) -> Optional[Tensor]:
        if not self.should_apply(modality, stage_idx):
            return None
        n_queries = queries.shape[1]
        assert patch_tokens.shape[1] == grid_size[0] * grid_size[1]
        # reuse the decoder's mask head + upsampling stack to generate the
        # query-specific attention regions
        mask_logits = self._mask_logits_fn(queries, patch_tokens, grid_size)
        mask_logits = F.interpolate(
            mask_logits,
            size=grid_size,
            mode='bilinear',
            align_corners=False
        )
        mask_flat = mask_logits.reshape(
            mask_logits.shape[0],
            n_queries,
            -1
        ) > 0
        patch_count = patch_tokens.shape[1]
        total_tokens = n_queries + n_prefix_tokens + patch_count
        attn_mask = torch.ones(
            queries.shape[0],
            total_tokens,
            total_tokens,
            dtype=torch.bool,
            device=queries.device
        )
        visual_start = n_queries + n_prefix_tokens
        attn_mask[:, :n_queries, visual_start:] = mask_flat
        attn_mask = self._apply_dropout(
            attn_mask,
            stage_idx,
            n_queries,
            visual_start
        )
        return attn_mask.unsqueeze(1)

    def _apply_dropout(
        self,
        attn_mask: Tensor,
        stage_idx: int,
        n_queries: int,
        visual_start: int
    ) -> Tensor:
        # branchless so torch.compile does not recompile per progress value.
        prob = self._probability_for_stage(stage_idx)
        rand = torch.rand(
            attn_mask.shape[0],
            n_queries,
            device=attn_mask.device
        )
        # in window -> unmask where rand > prob
        # after window -> unmask all queries
        # before window -> unmask none
        in_window = (prob > 0) & (prob < 1)
        in_post = prob <= 0
        query_mask = torch.where(
            in_window,
            rand > prob,
            in_post.expand_as(rand)
        )
        slab = attn_mask[:, :n_queries, visual_start:]
        slab |= query_mask.unsqueeze(-1)
        return attn_mask
