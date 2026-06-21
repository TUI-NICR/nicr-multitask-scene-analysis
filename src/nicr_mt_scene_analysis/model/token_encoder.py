# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn
from torch import Tensor

from ..types import EncoderForwardType
from ..types import EncoderInputType
from ..types import EncoderOutputType
from ..types import EncoderTokenInputType
from .activation import get_activation_class
from .attention_controller import AttentionMaskController
from .backbone.base import Backbone
from .backbone.base import TokenBackbone
from .encoder import ENCODER_META_KEY
from .encoder import Encoder
from .encoder import FusedRGBDEncoder
from .normalization import get_normalization_class
from .token_encoder_fusion import get_token_encoder_fusion_class


@dataclass
class _ExtraTokensState:
    # extra-token embeddings per modality. used when injecting
    # queries into the backbone stages for the first time.
    extra_tokens: Dict[str, Optional[Tensor]]
    masks: Dict[str, Tensor]
    injected: Set[str] = field(default_factory=set)
    processed_extra_tokens: Dict[str, Tensor] = field(default_factory=dict)
    # stores (patch_tokens, queries) pairs per (stage_idx, modality) so
    # decoder skip connections can reuse the exact tensors for
    # its heads.
    stage_token_cache: Dict[Tuple[int, str], Tuple[Tensor, Tensor]] = field(
        default_factory=dict
    )
    attention_mask_controller: Optional[AttentionMaskController] = None


class _TokenEncoderMixin:
    extra_tokens_key = '_extra_tokens'

    def __init__(
        self,
        *args,
        extra_tokens_start_stage_idx: int,
        extra_tokens_end_stage_idx: int,
        **kwargs
    ):
        self.extra_tokens_start_stage_idx = extra_tokens_start_stage_idx
        self.extra_tokens_end_stage_idx = extra_tokens_end_stage_idx
        super().__init__(*args, **kwargs)

        assert 0 <= extra_tokens_start_stage_idx
        assert extra_tokens_start_stage_idx <= extra_tokens_end_stage_idx
        assert extra_tokens_end_stage_idx < self._n_stages

    def _create_state(
        self,
        extra_tokens: Optional[EncoderTokenInputType],
        extra_tokens_attn_mask: Optional[EncoderInputType],
        extra_tokens_controller: Optional[AttentionMaskController]
    ) -> Optional[_ExtraTokensState]:
        tokens = dict(extra_tokens or {})
        if not any(value is not None for value in tokens.values()):
            return None
        masks = dict(extra_tokens_attn_mask or {})
        return _ExtraTokensState(
            extra_tokens=tokens,
            masks=masks,
            attention_mask_controller=extra_tokens_controller
        )

    def _ensure_token_backbone(self, backbone: Optional[Backbone], key: str):
        assert backbone is not None and isinstance(backbone, TokenBackbone), \
            f"Extra tokens for modality '{key}' require a token backbone."

    def _should_capture_stage(self, idx: int) -> bool:
        return (
            self.extra_tokens_start_stage_idx
            <= idx
            <= self.extra_tokens_end_stage_idx
        )

    def _controller_active(
        self,
        idx: int,
        controller: Optional[AttentionMaskController]
    ) -> bool:
        return (
            controller is not None
            and self.extra_tokens_start_stage_idx
            <= idx
            <= self.extra_tokens_end_stage_idx
        )

    def _cache_stage_tokens(
        self,
        state: Optional[_ExtraTokensState],
        idx: int,
        key: str,
        tokens: Tensor,
        queries: Tensor
    ) -> None:
        if state is None:
            return
        state.stage_token_cache[(idx, key)] = (tokens, queries)

    def _pop_stage_tokens(
        self,
        state: Optional[_ExtraTokensState],
        idx: int,
        key: str
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if state is None:
            return None, None
        return state.stage_token_cache.pop((idx, key), (None, None))

    def _split_normalized_tokens(
        self,
        *,
        backbone: TokenBackbone,
        idx: int,
        key: str,
        features: dict,
        extra_tokens: Tensor
    ) -> Optional[Tuple[Tensor, Tensor]]:
        total_queries = extra_tokens.shape[1]
        if total_queries <= 0 or key not in features:
            return None
        tokens_stage = features[key]
        if idx == self.extra_tokens_start_stage_idx:
            tokens_stage = backbone.concat_base_extra_tokens(
                tokens_stage,
                extra_tokens
            )
        tokens_norm = backbone.model.norm(tokens_stage)
        queries = tokens_norm[:, :total_queries, :]
        remainder = tokens_norm[:, total_queries:, :]
        return remainder, queries

    def _prepare_mask(
        self,
        state: Optional[_ExtraTokensState],
        key: str,
        sequence_length: int
    ) -> Optional[Tensor]:
        if not state or key not in state.masks:
            return None
        mask = state.masks[key]
        assert mask.shape[-2:] == (sequence_length, sequence_length)
        return mask

    def _stage_forward_kwargs(
        self,
        idx: int,
        key: str,
        backbones: Dict[str, Optional[Backbone]],
        state: Optional[_ExtraTokensState]
    ) -> Dict[str, Any]:
        if state is None:
            return {}
        in_window = (
            self.extra_tokens_start_stage_idx
            <= idx
            <= self.extra_tokens_end_stage_idx
        )
        if key not in state.injected or not in_window:
            return {}
        kwargs: Dict[str, Any] = {}
        if idx == self.extra_tokens_start_stage_idx:
            tokens = state.extra_tokens[key]
            backbone = backbones[key]
            self._ensure_token_backbone(backbone, key)
            kwargs['extra_tokens'] = tokens
        mask = state.masks.get(key)
        if mask is not None:
            kwargs['attn_mask'] = mask
        return kwargs

    def _pre_stage(
        self,
        idx: int,
        features: dict,
        backbones: dict,
        state: Optional[_ExtraTokensState]
    ) -> None:
        # runs before each backbone stage. inject the query (extra) tokens
        # on the first relevant stage, refresh the attention mask if the
        # controller asks for one, and snapshot the pre-stage tokens for any
        # decoder that wants this stage as a skip connection.
        if state is None:
            return

        for key, extra_tokens in state.extra_tokens.items():
            if key not in features or extra_tokens is None:
                continue
            n_extra = extra_tokens.shape[1]
            if n_extra == 0:
                continue
            backbone = backbones.get(key)
            self._ensure_token_backbone(backbone, key)
            sequence_length = features[key].shape[1]
            if idx < self.extra_tokens_start_stage_idx:
                sequence_length += n_extra
            mask = self._prepare_mask(state, key, sequence_length)
            if mask is not None:
                state.masks[key] = mask
            state.injected.add(key)

        capture_stage = self._should_capture_stage(idx)
        controller = state.attention_mask_controller
        controller_active = self._controller_active(idx, controller)
        if not capture_stage and not controller_active:
            return

        for key, extra_tokens in state.extra_tokens.items():
            if key not in features or extra_tokens is None:
                continue
            backbone = backbones[key]
            self._ensure_token_backbone(backbone, key)
            split = self._split_normalized_tokens(
                backbone=backbone,
                idx=idx,
                key=key,
                features=features,
                extra_tokens=extra_tokens
            )
            if split is None:
                continue
            remainder, queries = split
            if capture_stage:
                # cache pre-stage tokens and queries so decoder skip
                # connections can reuse them verbatim. this mirrors what EoMT
                # feeds into its auxiliary heads and keeps comparisons
                # bitwise-identical.
                self._cache_stage_tokens(state, idx, key, remainder, queries)

            if not controller_active or not controller.should_apply(key, idx):
                continue

            # meta can be for single backbone without encoder wrapper.
            meta = backbone.backbone_meta()
            # or it might be for Encoder/FusedRGBDEncoder
            if key in meta:
                meta = meta[key]

            n_prefix = meta['n_prefix_tokens']
            grid_size = meta['grid_size']
            patch_tokens = remainder[:, n_prefix:, :] if n_prefix else remainder

            mask = controller.build_mask(
                queries=queries,
                patch_tokens=patch_tokens,
                grid_size=grid_size,
                n_prefix_tokens=n_prefix,
                modality=key,
                stage_idx=idx
            )
            state.masks[key] = mask

    def _store_skip(
        self,
        idx: int,
        downsampling_factor: int,
        features: dict,
        decoder_skips_dict: dict,
        backbones: dict,
        state: Optional[_ExtraTokensState]
    ) -> None:
        ds_key = str(downsampling_factor)
        injected = state.injected if state else set()
        should_strip = (
            state is not None
            and injected
            and self.extra_tokens_start_stage_idx
            <= idx
            <= self.extra_tokens_end_stage_idx
        )
        if should_strip:
            base_features: Dict[str, Optional[Tensor]] = {}
            extracted_tokens: Dict[str, Tensor] = {}
            for key, tensor in features.items():
                if key in injected:
                    backbone = backbones.get(key)
                    self._ensure_token_backbone(backbone, key)
                    base_tokens, pulled = self._pop_stage_tokens(
                        state, idx, key
                    )
                    if base_tokens is None and pulled is None:
                        # no cached tensors means extra tokens were not active
                        # at this stage. fall back to extracting the standard
                        # post-stage tensors to preserve prior behavior.
                        base_tokens, pulled = (
                            backbone.extract_active_extra_tokens(tensor)
                        )
                        if base_tokens is not None:
                            base_tokens = backbone.finalize_head_tokens(
                                base_tokens
                            )
                        if pulled is not None:
                            pulled = backbone.finalize_head_tokens(pulled)
                    base_features[key] = base_tokens
                    if pulled is not None:
                        extracted_tokens[key] = pulled
                else:
                    base_features[key] = tensor

            entry = decoder_skips_dict.setdefault(ds_key, {})
            entry.update(base_features)

            if extracted_tokens:
                extra_dict = entry.get(self.extra_tokens_key)
                if extra_dict is None:
                    extra_dict = {}
                    entry[self.extra_tokens_key] = extra_dict
                extra_dict.update(extracted_tokens)
        else:
            entry = decoder_skips_dict.setdefault(ds_key, {})
            entry.update(features)

        # token decoders read this to map skips back to encoder stages
        entry['_stage_idx'] = idx

    def _post_stage(
        self,
        idx: int,
        features: dict,
        backbones: dict,
        state: Optional[_ExtraTokensState]
    ) -> None:
        # runs after each backbone stage. once we hit the end stage of the
        # extra-token range, pull the queries back out of the sequence so the
        # remaining stages run on the original patch tokens only.
        if state is None or idx != self.extra_tokens_end_stage_idx:
            return

        for key in list(state.injected):
            if key not in features:
                continue
            backbone = backbones.get(key)
            self._ensure_token_backbone(backbone, key)
            base_tokens, extracted = backbone.extract_active_extra_tokens(
                features[key]
            )
            features[key] = base_tokens
            if extracted is not None:
                state.processed_extra_tokens[key] = extracted
            state.injected.discard(key)
            backbone.clear_active_extra_tokens()
            state.masks.pop(key, None)

    def _collect_encoder_meta(
        self,
        backbones: Dict[str, Optional[Backbone]]
    ) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        for key, backbone in backbones.items():
            if backbone is None or not hasattr(backbone, 'backbone_meta'):
                continue
            info = backbone.backbone_meta()
            if info:
                meta[key] = info
        return meta

    def _build_encoder_output(
        self,
        x_: Dict[str, Tensor],
        backbones: Dict[str, Optional[Backbone]],
        decoder_skips_dict: Dict[str, dict],
        state: Optional[_ExtraTokensState]
    ) -> EncoderOutputType:
        # assemble the (features, skips, [extras]) return tuple shared by every
        # concrete forward in this module.
        features: Dict[str, Any] = dict(x_)
        meta = self._collect_encoder_meta(backbones)
        if meta:
            features[ENCODER_META_KEY] = meta
        skips = dict(decoder_skips_dict)
        # return the 3-tuple whenever a token state exists, even if it carries
        # no processed tokens (token_num_queries == 0). this keeps the encoder
        # output shape consistent for token tasks that do not consume queries.
        if state is not None:
            return features, skips, dict(state.processed_extra_tokens)
        return features, skips


class TokenEncoder(_TokenEncoderMixin, Encoder):
    def forward(
        self,
        x: EncoderInputType,
        extra_tokens: Optional[EncoderTokenInputType] = None,
        extra_tokens_attn_mask: Optional[EncoderInputType] = None,
        extra_tokens_controller: Optional[AttentionMaskController] = None
    ) -> EncoderOutputType:
        state = self._create_state(
            extra_tokens, extra_tokens_attn_mask, extra_tokens_controller
        )

        assert len(x) == 1
        key = list(x.keys())[0]
        x_ = {key: x[key]}
        backbones = {key: self.backbone}
        decoder_skips_dict: Dict[str, dict] = {}

        downsampling_idx = 0
        for idx in range(self._n_stages):
            self._pre_stage(idx, x_, backbones, state)
            stage_kwargs = self._stage_forward_kwargs(
                idx, key, backbones, state
            )
            x_[key] = self.backbone.forward_stage(
                idx, x_[key], **stage_kwargs
            )
            if self._stages_skip_connections[idx]:
                cur_downsampling = self.skips_downsamplings[downsampling_idx]
                self._store_skip(
                    idx,
                    cur_downsampling,
                    x_,
                    decoder_skips_dict,
                    backbones,
                    state
                )
                downsampling_idx += 1
            self._post_stage(idx, x_, backbones, state)

        return self._build_encoder_output(
            x_, backbones, decoder_skips_dict, state
        )


class FusedTokenEncoder(_TokenEncoderMixin, FusedRGBDEncoder):
    def __init__(
        self,
        backbone_rgb: Optional[Backbone],
        backbone_depth: Optional[Backbone],
        fusion,
        normalization=None,
        activation=None,
        skip_downsamplings: Sequence[int] = (4, 8, 16),
        fusion_stage_indices: Optional[Sequence[int]] = None,
        *,
        extra_tokens_start_stage_idx: int,
        extra_tokens_end_stage_idx: int
    ):
        super().__init__(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            fusion=fusion,
            normalization=normalization,
            activation=activation,
            skip_downsamplings=skip_downsamplings,
            extra_tokens_start_stage_idx=extra_tokens_start_stage_idx,
            extra_tokens_end_stage_idx=extra_tokens_end_stage_idx
        )

        # FusedRGBDEncoder built fusion modules for every stage. when the
        # caller restricts fusion to a subset, rebuild self.fusions and the
        # idx -> module mapping so forward() can dispatch by stage index.
        if fusion_stage_indices is None:
            self._fusion_stage_indices = tuple(range(self._n_stages))
        else:
            self._fusion_stage_indices = self._prepare_fusion_stage_indices(
                fusion_stage_indices
            )
        self._fusion_stage_set = set(self._fusion_stage_indices)
        if backbone_rgb is not None and backbone_depth is not None:
            if fusion_stage_indices is None:
                self._fusion_stage_to_module = {
                    idx: module for idx, module in enumerate(self.fusions)
                }
            else:
                self._fusion_stage_to_module = self._rebuild_subset_fusions(
                    fusion, normalization, activation
                )
        else:
            self._fusion_stage_to_module = {}

    def _prepare_fusion_stage_indices(
        self,
        indices: Sequence[int]
    ) -> Tuple[int, ...]:
        cleaned = tuple(dict.fromkeys(indices))
        assert all(0 <= idx < self._n_stages for idx in cleaned), \
            'fusion_stage_indices must be within ' \
            f'[0, {self._n_stages - 1}] but got {cleaned}.'
        return cleaned

    def _rebuild_subset_fusions(self, fusion, normalization, activation):
        # replace the all-stages fusion modules built by FusedRGBDEncoder
        # with only the requested subset, so checkpoints map cleanly.
        b_rgb = self.backbone_rgb
        modules = []
        idx_to_module = {}
        for idx in self._fusion_stage_indices:
            n = b_rgb.stages_n_channels[idx]
            memory_layout = b_rgb.stages_memory_layout[idx]
            module = fusion(
                n_channels_in=n,
                normalization=normalization,
                activation=activation,
                input_memory_layout=memory_layout
            )
            modules.append(module)
            idx_to_module[idx] = module
        self.fusions = nn.ModuleList(modules)
        return idx_to_module

    def _split_for_fusion(
        self,
        x: EncoderForwardType
    ) -> Tuple[EncoderForwardType, Dict[str, Optional[Tensor]]]:
        # split off prefix/extra tokens so the per-stage fusion module only
        # operates on the visual tokens that exist for both modalities.
        x_rgb = x['rgb']
        x_depth = x['depth']
        extras: Dict[str, Optional[Tensor]] = {'rgb': None, 'depth': None}
        if x_rgb.dim() != 3 or x_depth.dim() != 3:
            assert x_rgb.shape == x_depth.shape, (
                "RGB-D fusion expects matching shapes unless extra tokens are "
                "present."
            )
            return x, extras
        len_rgb = x_rgb.shape[1]
        len_depth = x_depth.shape[1]
        if len_rgb == len_depth:
            return x, extras
        base_len = min(len_rgb, len_depth)
        assert base_len > 0
        if len_rgb > base_len:
            extras['rgb'] = x_rgb[:, :len_rgb - base_len, :]
        if len_depth > base_len:
            extras['depth'] = x_depth[:, :len_depth - base_len, :]
        base = {
            'rgb': x_rgb[:, -base_len:, :],
            'depth': x_depth[:, -base_len:, :],
        }
        return base, extras

    def _merge_fusion(
        self,
        fused: EncoderForwardType,
        extras: Dict[str, Optional[Tensor]]
    ) -> EncoderForwardType:
        rgb_extra = extras.get('rgb')
        depth_extra = extras.get('depth')
        if rgb_extra is not None:
            fused['rgb'] = torch.cat([rgb_extra, fused['rgb']], dim=1)
        if depth_extra is not None:
            fused['depth'] = torch.cat([depth_extra, fused['depth']], dim=1)
        return fused

    def forward(
        self,
        x: EncoderInputType,
        extra_tokens: Optional[EncoderTokenInputType] = None,
        extra_tokens_attn_mask: Optional[EncoderInputType] = None,
        extra_tokens_controller: Optional[AttentionMaskController] = None
    ) -> EncoderOutputType:
        assert len(x) == 2

        state = self._create_state(
            extra_tokens, extra_tokens_attn_mask, extra_tokens_controller
        )

        backbone_rgb = self.backbone_rgb
        backbone_depth = self.backbone_depth
        x_: Dict[str, Tensor] = {'rgb': x['rgb'], 'depth': x['depth']}
        backbones: Dict[str, Optional[Backbone]] = {
            'rgb': backbone_rgb, 'depth': backbone_depth
        }
        decoder_skips_dict: Dict[str, dict] = {}

        downsampling_idx = 0
        for idx in range(self._n_stages):
            self._pre_stage(idx, x_, backbones, state)

            if backbone_rgb is not None:
                kwargs_rgb = self._stage_forward_kwargs(
                    idx, 'rgb', backbones, state
                )
                x_['rgb'] = backbone_rgb.forward_stage(
                    idx, x_['rgb'], **kwargs_rgb
                )
            if backbone_depth is not None:
                kwargs_depth = self._stage_forward_kwargs(
                    idx, 'depth', backbones, state
                )
                x_['depth'] = backbone_depth.forward_stage(
                    idx, x_['depth'], **kwargs_depth
                )

            if idx in self._fusion_stage_set:
                fusion_module = self._fusion_stage_to_module[idx]
                fusion_inputs, extras = self._split_for_fusion(x_)
                fused = fusion_module(fusion_inputs)
                x_ = self._merge_fusion(fused, extras)

            if self._stages_skip_connections[idx]:
                cur_downsampling = self.skips_downsamplings[downsampling_idx]
                self._store_skip(
                    idx,
                    cur_downsampling,
                    x_,
                    decoder_skips_dict,
                    backbones,
                    state
                )
                downsampling_idx += 1

            self._post_stage(idx, x_, backbones, state)

        return self._build_encoder_output(
            x_, backbones, decoder_skips_dict, state
        )


TokenEncoderType = Union[TokenEncoder, FusedTokenEncoder]


def get_token_encoder(
    backbone_rgb: Optional[Backbone] = None,
    backbone_depth: Optional[Backbone] = None,
    backbone_rgbd: Optional[Backbone] = None,
    fusion: Optional[str] = None,
    fusion_stage_indices: Optional[Sequence[int]] = None,
    normalization: str = 'batchnorm',
    activation: str = 'relu',
    skip_downsamplings: Sequence[int] = (4, 8, 16),
    *,
    extra_tokens_start_stage_idx: int,
    extra_tokens_end_stage_idx: int
) -> TokenEncoderType:
    if backbone_rgb is not None and backbone_depth is not None:
        return FusedTokenEncoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            fusion=get_token_encoder_fusion_class(fusion),
            normalization=get_normalization_class(normalization),
            activation=get_activation_class(activation),
            skip_downsamplings=skip_downsamplings,
            fusion_stage_indices=fusion_stage_indices,
            extra_tokens_start_stage_idx=extra_tokens_start_stage_idx,
            extra_tokens_end_stage_idx=extra_tokens_end_stage_idx
        )

    if fusion_stage_indices is not None:
        raise ValueError(
            'fusion_stage_indices can only be used for RGB-D fusion.'
        )
    if backbone_rgbd is not None:
        backbone = backbone_rgbd
    elif backbone_rgb is not None:
        backbone = backbone_rgb
    elif backbone_depth is not None:
        backbone = backbone_depth
    else:
        raise ValueError(
            'Either `backbone_rgb` and/or `backbone_depth` or '
            '`backbone_rgbd` must be given.'
        )

    return TokenEncoder(
        backbone=backbone,
        skip_downsamplings=skip_downsamplings,
        extra_tokens_start_stage_idx=extra_tokens_start_stage_idx,
        extra_tokens_end_stage_idx=extra_tokens_end_stage_idx,
    )
