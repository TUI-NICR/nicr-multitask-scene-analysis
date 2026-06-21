# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import abc

import torch
from torch import Tensor
import torch.nn as nn


class Backbone(abc.ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def stages(self) -> List[Union[nn.Sequential, nn.Module]]:
        pass

    @property
    @abc.abstractmethod
    def stages_n_channels(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def stages_downsampling(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def stages_memory_layout(self) -> List[str]:
        pass

    def forward_stage(self, stage_idx: int, x: Tensor) -> Tensor:
        stage = self.stages[stage_idx]
        return stage(x)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.stages)):
            x = self.forward_stage(i, x)
        return x


class TokenBackbone(Backbone, abc.ABC):

    def __init__(self, n_base_prefix_tokens: int = 0):
        super().__init__()
        self._n_base_prefix_tokens = n_base_prefix_tokens
        # length of the extras currently prepended to the token sequence.
        # set in _maybe_activate_extra_tokens, cleared after extraction.
        self._active_extra_token_length = 0

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        pass

    @abc.abstractmethod
    def set_input_size(
        self,
        height: int,
        width: int,
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None
    ):
        # resize the patch embedding / positional embedding for the given
        # image and patch size.
        pass

    def concat_base_extra_tokens(
        self,
        tokens: torch.Tensor,
        extra_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if extra_tokens is None:
            return tokens

        assert extra_tokens.dim() == 3, \
            "extra_tokens must use BND layout."
        assert extra_tokens.shape[-1] == tokens.shape[-1], (
            f"extra_tokens dim {extra_tokens.shape[-1]} != "
            f"token dim {tokens.shape[-1]}."
        )
        assert extra_tokens.shape[0] == tokens.shape[0], (
            f"extra_tokens batch {extra_tokens.shape[0]} != {tokens.shape[0]}."
        )

        # prepend extra tokens so they stay before timm's prefix tokens.
        injected = torch.cat([extra_tokens, tokens], dim=1)
        return injected

    def split_base_extra_tokens(
        self,
        tokens: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if k == 0:
            return tokens, None
        extra = tokens[:, :k, :]
        base = tokens[:, k:, :]
        return base, extra

    def forward_stage(
        self,
        stage_idx: int,
        x: Tensor,
        *,
        extra_tokens: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # forward one transformer block, optionally injecting extra prefix
        # tokens and propagating an attention mask.
        x = self._maybe_activate_extra_tokens(x, extra_tokens)
        self._on_extra_tokens_activated()
        return self._forward_stage_impl(
            stage_idx,
            x,
            attn_mask=attn_mask,
            **kwargs
        )

    @abc.abstractmethod
    def _forward_stage_impl(
        self,
        stage_idx: int,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        pass

    def _on_extra_tokens_activated(self) -> None:
        # hook for RoPE-aware subclasses to sync attention submodules.
        pass

    def _maybe_activate_extra_tokens(
        self,
        tensor: Tensor,
        extra_tokens: Optional[Tensor]
    ) -> Tensor:
        if extra_tokens is None:
            return tensor

        length = extra_tokens.shape[1]
        tensor_with_extra = self.concat_base_extra_tokens(
            tensor,
            extra_tokens
        )

        self._active_extra_token_length = length
        return tensor_with_extra

    def _current_extra_token_count(self) -> int:
        return self._active_extra_token_length

    def clear_active_extra_tokens(self) -> None:
        self._active_extra_token_length = 0

    def extract_active_extra_tokens(
        self,
        tokens: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        total = self._current_extra_token_count()
        if total == 0:
            return tokens, None

        base, extracted = self.split_base_extra_tokens(tokens, total)
        return base, extracted

    @abc.abstractmethod
    def backbone_meta(self) -> Dict[str, Any]:
        pass

    def finalize_head_tokens(self, tokens: Tensor) -> Tensor:
        return tokens
