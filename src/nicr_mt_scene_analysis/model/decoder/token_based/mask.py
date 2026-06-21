# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ....types import DecoderRawOutputType
from ....types import EncoderSkipsType
from ....types import TokenDecoderInputType
from ...postprocessing import get_postprocessing_class
from ...postprocessing import PostProcessingType
from ...upsampling import get_upsampling_class
from .base import TokenDecoderBase


class TokenMaskDecoder(TokenDecoderBase):
    # mask head following EoMT (Kerssies et al., CVPR 2025): an MLP maps
    # each query to a mask embedding that is combined with the upsampled patch
    # features to produce per-query mask logits (mask classification as in
    # MaskFormer, Cheng et al., 2021).
    def __init__(
        self,
        *,
        embed_dim: int,
        n_upsampling_blocks: int = 2,
        modality: Optional[str] = None,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class(
            'token-mask'
        ),
        upsampling_mode: str = 'transposed-conv',
        prediction_upsampling: str = 'bilinear',
        upsample_side_outputs: bool = True,
        side_output_stage_indices: Optional[Tuple[int, ...]] = None
    ):
        super().__init__(embed_dim=embed_dim,
                         modality=modality,
                         postprocessing=postprocessing,
                         side_output_stage_indices=side_output_stage_indices)

        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        if n_upsampling_blocks < 0:
            raise ValueError("n_upsampling_blocks must be non-negative.")
        # Each learned block upsamples patch features by 2. The caller chooses
        # the number of blocks from the backbone patch size. For patch size 16,
        # two learned blocks map the patch grid to 1/4 input resolution. Patch
        # size 14 cannot be reconstructed exactly by chained 2x blocks and may
        # require an additional interpolation step before full-resolution
        # postprocessing.
        n_upscale = n_upsampling_blocks
        upsampling_cls = get_upsampling_class(
            upsampling_mode,
            n_channels=embed_dim,
            scale_factor=2.
        )
        self.upscale = nn.Sequential(*[
            upsampling_cls()
            for _ in range(n_upscale)
        ])
        self._prediction_upsampling = prediction_upsampling
        self._upsample_side_outputs = upsample_side_outputs

    def _resize_prediction(
        self,
        output: torch.Tensor,
        shape: Tuple[int, int]
    ) -> torch.Tensor:
        if output.shape[-2:] == shape:
            return output
        mode = (
            self._prediction_upsampling
            if self._prediction_upsampling in ('nearest', 'bilinear')
            else 'bilinear'
        )
        kwargs = {'align_corners': False} if mode == 'bilinear' else {}
        return interpolate(output, size=shape, mode=mode, **kwargs)

    def compute_mask_logits(
        self,
        queries: torch.Tensor,
        patch_tokens: torch.Tensor,
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        grid_h, grid_w = grid_size
        # reshape [B, N, D] -> [B, D, H, W]
        patches = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            self.embed_dim,
            grid_h,
            grid_w
        )
        mask_features = self.upscale(patches)
        mask_queries = self.mask_head(queries)
        # mask_queries is [B, Q, D] and mask_features is [B, D, H, W].
        # The einsum computes the dot product along D for each query and each
        # spatial feature vector. The result is [B, Q, H, W]: Q mask logit maps
        # of shape HxW per batch sample.
        return torch.einsum('bqd,bdhw->bqhw', mask_queries, mask_features)

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> DecoderRawOutputType:
        tokens_by_modality, _, extra_tokens = x
        key = self._select_modality(tokens_by_modality)
        modality_meta = tokens_by_modality['_encoder_meta'][key]
        grid_size = modality_meta['grid_size']
        prefix_tokens = modality_meta.get('n_prefix_tokens', 0)
        tokens = tokens_by_modality[key]
        # strip prefix tokens before restoring the spatial layout.
        if prefix_tokens:
            tokens = tokens[:, prefix_tokens:, :]
        queries = self._collect_queries(extra_tokens, key)
        output = self.compute_mask_logits(queries, tokens, grid_size)

        side_outputs = []
        for entry in self._iter_side_entries(skips):
            if '_extra_tokens' not in entry or key not in entry:
                continue
            x_stage = entry[key]
            if prefix_tokens:
                x_stage = x_stage[:, prefix_tokens:, :]
            queries_aux = self._collect_queries(entry['_extra_tokens'], key)
            output_skip = self.compute_mask_logits(
                queries_aux, x_stage, grid_size
            )
            side_outputs.append(output_skip)
        return output, tuple(side_outputs)

    def _forward_inference(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> DecoderRawOutputType:
        tokens_by_modality, _, _ = x
        key = self._select_modality(tokens_by_modality)
        modality_meta = tokens_by_modality['_encoder_meta'][key]
        output, side_outputs = self._forward_training(x, skips)
        # Training losses sample points from the decoder-resolution masks.
        # Inference upsamples masks to the input size from encoder meta before
        # semantic/panoptic postprocessing.
        patch_size = modality_meta.get('patch_size')
        if patch_size is None:
            return output, side_outputs
        grid_h, grid_w = modality_meta['grid_size']
        patch_h, patch_w = patch_size
        target_shape = (grid_h * patch_h, grid_w * patch_w)
        output = self._resize_prediction(output, target_shape)
        if self._upsample_side_outputs:
            side_outputs = tuple(
                self._resize_prediction(aux, target_shape)
                for aux in (side_outputs or ())
            )
        return output, side_outputs
