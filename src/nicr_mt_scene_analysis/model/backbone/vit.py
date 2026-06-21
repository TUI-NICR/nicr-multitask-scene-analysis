# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Benedict Stephan <benedict.stephan@tu-ilmenau.de>
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from timm.layers import Format
from timm.layers import PatchEmbed
from timm.layers import nchw_to
from timm.layers import resample_patch_embed
from timm.layers import to_2tuple
from timm.models import adapt_input_conv
from timm.models import load_pretrained
from timm.models.eva import EvaAttention
from torch import Tensor
from torch.nn import functional as F

from nicr_mt_scene_analysis.model.backbone import TokenBackbone


class RGBDProjection(nn.Module):
    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = (224, 244),
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        embed_dim: int = 768,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()

        device_and_dtype_dict = {'device': device, 'dtype': dtype}

        self.patch_size = patch_size

        # separate convs per modality so RGB can keep pretrained patch-embed
        # weights and depth gets its own learnable projection.
        self.rgb_proj = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **device_and_dtype_dict
        )
        self.depth_proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **device_and_dtype_dict
        )

    @property
    def weight(self) -> Tensor:
        # concatenated [RGB | depth] kernel, useful when external code wants
        # to inspect the equivalent 4-channel projection.
        return torch.cat([self.rgb_proj.weight, self.depth_proj.weight], dim=1)

    @torch.no_grad()
    def set_patch_size(self, patch_size: Tuple[int, int]) -> None:
        rgb_proj_new = nn.Conv2d(
            self.rgb_proj.in_channels,
            self.rgb_proj.out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=self.rgb_proj.bias is not None,
            device=self.rgb_proj.weight.device,
            dtype=self.rgb_proj.weight.dtype
        )
        rgb_proj_new.weight.copy_(
            resample_patch_embed(
                self.rgb_proj.weight,
                patch_size,
                verbose=True
            )
        )
        if self.rgb_proj.bias is not None:
            rgb_proj_new.bias.copy_(self.rgb_proj.bias)
        self.rgb_proj = rgb_proj_new

        depth_proj_new = nn.Conv2d(
            self.depth_proj.in_channels,
            self.depth_proj.out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=self.depth_proj.bias is not None,
            device=self.depth_proj.weight.device,
            dtype=self.depth_proj.weight.dtype
        )
        depth_proj_new.weight.copy_(
            resample_patch_embed(
                self.depth_proj.weight,
                patch_size,
                verbose=True
            )
        )
        if self.depth_proj.bias is not None:
            depth_proj_new.bias.copy_(self.depth_proj.bias)
        self.depth_proj = depth_proj_new

        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_x = x[:, :3, :, :]
        depth_x = x[:, 3:, :, :]

        rgb_embeddings = self.rgb_proj(rgb_x)
        depth_embeddings = self.depth_proj(depth_x)

        embeddings = rgb_embeddings + depth_embeddings

        return embeddings


class CustomRGBDPatchEmbed(PatchEmbed):
    # RGB-D variant of timm's PatchEmbed. Plain ViTs receive it via
    # `embed_layer`. EVA-style factories need a temporary PatchEmbed swap.
    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        assert in_chans == 4, "CustomRGBDPatchEmbed requires RGB-D input."
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
            strict_img_size=strict_img_size,
            dynamic_img_pad=dynamic_img_pad,
            device=device,
            dtype=dtype
        )
        self.proj = RGBDProjection(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> None:
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            self.proj.set_patch_size(new_patch_size)
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = \
                self._init_img_size(img_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.dynamic_img_pad:
            height, width = x.shape[-2:]
            patch_h, patch_w = self.patch_size
            pad_h = (patch_h - height % patch_h) % patch_h
            pad_w = (patch_w - width % patch_w) % patch_w
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class _TimmViTBackbone(TokenBackbone):
    # shared base for plain timm ViT backbones (DeiT3, DINOv2).
    # subclasses override `_default_patch_size`.
    _default_patch_size: int = 16

    def __init__(self, model: nn.Module, n_input_channels: int = 3):
        super().__init__(n_base_prefix_tokens=model.num_prefix_tokens)
        self.model = model
        self.n_input_channels = n_input_channels
        self._embed_dim = model.embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def stages(self) -> List[Union[nn.Sequential, nn.Module]]:
        return [self.model.patch_embed, *list(self.model.blocks)]

    @property
    def stages_n_channels(self) -> List[int]:
        return [self.embed_dim] * len(self.stages)

    @property
    def stages_downsampling(self) -> List[int]:
        # use stage idx as downsampling factor for skip-connection bookkeeping
        return list(range(len(self.stages)))

    @property
    def stages_memory_layout(self) -> List[str]:
        # token features use BND layout (batch, token, feature dimension)
        return ['BND'] * len(self.stages)

    def set_input_size(
        self,
        height: int,
        width: int,
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None
    ) -> None:
        patch_height = patch_height or self._default_patch_size
        patch_width = patch_width or self._default_patch_size
        self.model.set_input_size(
            img_size=(height, width),
            patch_size=(patch_height, patch_width)
        )

    def backbone_meta(self) -> Dict[str, Any]:
        patch = tuple(int(v) for v in self.model.patch_embed.patch_size)
        grid = tuple(int(v) for v in self.model.patch_embed.grid_size)
        return {
            'active_extra_tokens': self._current_extra_token_count(),
            'patch_size': patch,
            'grid_size': grid,
            'n_prefix_tokens': self._n_base_prefix_tokens,
            'embed_dim': self.embed_dim,
        }

    def _forward_stage_impl(
        self,
        stage_idx: int,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        # stage 0 runs the patch embed + positional embedding without a
        # transformer block. stages 1..N each run one transformer block.
        last_stage_idx = len(self.model.blocks)

        if stage_idx == 0:
            tokens = self.model.patch_embed(x)
            tokens = self.model._pos_embed(tokens)
            tokens = self.model.patch_drop(tokens)
            tokens = self.model.norm_pre(tokens)
            return tokens

        tokens = self.model.blocks[stage_idx - 1](x, attn_mask=attn_mask)

        if stage_idx == last_stage_idx:
            # use the final encoder norm only after the last block.
            # intermediate stages expose the block outputs directly.
            tokens = self.model.norm(tokens)

        return tokens

    def finalize_head_tokens(self, tokens: Tensor) -> Tensor:
        return self.model.norm(tokens)


class _RopePrefixTokensMixin:
    # EVA-style timm blocks use rotary position embeddings (RoPE) only for
    # tokens with a 2D image-grid position. CLS/register tokens and our
    # injected extra tokens have no such grid position, so timm splits Q/K into
    # prefix tokens and patch tokens before applying RoPE. The split point is
    # stored as the integer EvaAttention.num_prefix_tokens. Since our extra
    # tokens are prepended in front of timm's own CLS/register tokens, this
    # value must be updated while extra tokens are active.
    def _collect_rope_prefix_modules(self) -> List[EvaAttention]:
        modules = [
            m for m in self.model.modules()
            if isinstance(m, EvaAttention)
        ]
        for module in modules:
            if not hasattr(module, 'num_prefix_tokens'):
                raise AttributeError(
                    "timm EvaAttention no longer exposes "
                    "`num_prefix_tokens`."
                )
        return modules

    def _on_extra_tokens_activated(self) -> None:
        value = self._n_base_prefix_tokens + self._active_extra_token_length
        for m in self._rope_prefix_modules:
            m.num_prefix_tokens = value


class _EvaStyleBackbone(_RopePrefixTokensMixin, _TimmViTBackbone):
    # shared base for EVA-style backbones (DINOv3, EVA-02) with rope plumbing.
    def __init__(self, model: nn.Module, n_input_channels: int = 3):
        super().__init__(model, n_input_channels)
        self._rope_prefix_modules: List[EvaAttention] = \
            self._collect_rope_prefix_modules()

    def _forward_stage_impl(
        self,
        stage_idx: int,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        # same stage-0 vs stage-N layout as the plain ViT, plus rope tensors
        # produced at stage 0 are stashed on self and threaded through every
        # block (some EVA models even return per-stage rope tuples).
        last_stage_idx = len(self.model.blocks)

        if stage_idx == 0:
            tokens = self.model.patch_embed(x)
            tokens, rope = self.model._pos_embed(tokens)
            tokens = self.model.norm_pre(tokens)
            # store rope for later blocks. rope is only applied to visual
            # tokens. extras sit in front of the prefix and are skipped.
            self._rope = rope
            return tokens

        rope = self._rope
        # rope_mixed: per-stage rope tuples (only set by some Eva models).
        if getattr(self.model, 'rope_mixed', False) and rope is not None:
            rope_idx = stage_idx - 1
            if rope_idx < len(rope):
                rope = rope[rope_idx]
        tokens = self.model.blocks[stage_idx - 1](
            x,
            rope=rope,
            attn_mask=attn_mask
        )

        if stage_idx == last_stage_idx:
            # use the final encoder norm only after the last block.
            # intermediate stages expose the block outputs directly.
            tokens = self.model.norm(tokens)

        return tokens


class DeiT3Backbone(_TimmViTBackbone):
    _default_patch_size = 16


class DinoV2Backbone(_TimmViTBackbone):
    _default_patch_size = 14


class Eva02Backbone(_EvaStyleBackbone):
    _default_patch_size = 14


class DinoV3Backbone(_EvaStyleBackbone):
    _default_patch_size = 16


def _split_rgbd_patch_projection_state_dict(
    state_dict: Dict[str, Tensor],
    model: nn.Module
) -> Dict[str, Tensor]:
    proj = model.patch_embed.proj
    if not isinstance(proj, RGBDProjection):
        return state_dict

    weight = state_dict.pop('patch_embed.proj.weight')
    adjusted = adapt_input_conv(4, weight)
    state_dict['patch_embed.proj.rgb_proj.weight'] = adjusted[:, :3, :, :]
    state_dict['patch_embed.proj.depth_proj.weight'] = adjusted[:, 3:, :, :]

    bias = state_dict.pop('patch_embed.proj.bias', None)
    if bias is not None:
        state_dict['patch_embed.proj.rgb_proj.bias'] = bias
        state_dict['patch_embed.proj.depth_proj.bias'] = bias

    return state_dict


def _load_pretrained_rgb_weights(
    model: nn.Module,
    n_input_channels: int
) -> None:
    filter_fn = None
    load_in_chans = n_input_channels

    if isinstance(model.patch_embed.proj, RGBDProjection):
        assert n_input_channels == 4, (
            "RGBDProjection only supports n_input_channels=4"
        )
        filter_fn = _split_rgbd_patch_projection_state_dict
        # let the filter split the RGB weights into RGB and depth projections.
        load_in_chans = 3

    load_pretrained(
        model,
        pretrained_cfg=model.pretrained_cfg,
        num_classes=getattr(model, 'num_classes', 1000),
        in_chans=load_in_chans,
        filter_fn=filter_fn,
        strict=False
    )


def _create_model_with_vit_patch_embed_override(
    timm_model_name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> nn.Module:
    # VisionTransformer accepts `embed_layer` as a kwarg.
    create_kwargs = dict(kwargs)
    if n_input_channels == 4:
        create_kwargs['embed_layer'] = CustomRGBDPatchEmbed
    model = timm.create_model(
        timm_model_name,
        pretrained=pretrained and (n_input_channels == 3),
        in_chans=n_input_channels,
        **create_kwargs
    )
    if pretrained and n_input_channels != 3:
        _load_pretrained_rgb_weights(
            model,
            n_input_channels
        )
    return model


def _create_model_with_eva_patch_embed_override(
    timm_model_name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> nn.Module:
    # EVA creates PatchEmbed inside the model factory and does not expose an
    # embed_layer argument like the plain ViT factory. For RGB-D input, replace
    # the module attribute only while the model is being constructed. Always
    # restore the original class so a failed model creation cannot leak the
    # RGB-D replacement into later calls.
    original_patch_embed = timm.models.eva.PatchEmbed
    try:
        if n_input_channels == 4:
            timm.models.eva.PatchEmbed = CustomRGBDPatchEmbed
        model = timm.create_model(
            timm_model_name,
            pretrained=pretrained and (n_input_channels == 3),
            in_chans=n_input_channels,
            **kwargs
        )
    finally:
        timm.models.eva.PatchEmbed = original_patch_embed

    if pretrained and n_input_channels != 3:
        _load_pretrained_rgb_weights(
            model,
            n_input_channels
        )
    return model


def get_deit3_backbone(
    name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> TokenBackbone:
    name = name.lower()
    if not name.startswith('deit3_'):
        raise ValueError(f"Invalid DeiT3 backbone name: '{name}'")

    model = _create_model_with_vit_patch_embed_override(
        name, pretrained, n_input_channels, **kwargs
    )
    return DeiT3Backbone(model=model, n_input_channels=n_input_channels)


def get_dinov2_backbone(
    name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> TokenBackbone:
    name = name.lower()
    if not name.startswith('dinov2_'):
        raise ValueError(f"Invalid DINOv2 backbone name: '{name}'")

    # expected names like:
    #   dinov2_small, dinov2_base, dinov2_large, dinov2_giant
    # and optional register tokens:
    #   dinov2_small_reg4, dinov2_base_reg4, ...
    rest = name[len('dinov2_'):]
    reg4 = rest.endswith('_reg4')
    if reg4:
        rest = rest[:-len('_reg4')]
    size = rest

    # timm naming convention used by DINOv2 ViTs
    # examples:
    #   vit_large_patch14_dinov2.lvd142m
    #   vit_large_patch14_reg4_dinov2.lvd142m
    reg_part = 'reg4_' if reg4 else ''
    timm_model_name = f'vit_{size}_patch14_{reg_part}dinov2.lvd142m'

    model = _create_model_with_vit_patch_embed_override(
        timm_model_name, pretrained, n_input_channels, **kwargs
    )
    return DinoV2Backbone(model=model, n_input_channels=n_input_channels)


def get_eva02_backbone(
    name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> TokenBackbone:
    name = name.lower()
    if not name.startswith('eva02_'):
        raise ValueError(f"Invalid EVA02 backbone name: '{name}'")

    model = _create_model_with_eva_patch_embed_override(
        name, pretrained, n_input_channels, **kwargs
    )
    return Eva02Backbone(model=model, n_input_channels=n_input_channels)


def get_dinov3_backbone(
    name: str,
    pretrained: bool,
    n_input_channels: int,
    **kwargs: Any
) -> TokenBackbone:
    name = name.lower()
    if not name.startswith('dinov3_'):
        raise ValueError(f"Invalid DINOv3 backbone name: '{name}'")

    # expected names like:
    #   dinov3_tiny, dinov3_small, dinov3_base, dinov3_large
    # and optional QKV bias:
    #   dinov3_small_qkvb, dinov3_base_qkvb, ...
    rest = name[len('dinov3_'):]
    qkvb = rest.endswith('_qkvb')
    if qkvb:
        rest = rest[:-len('_qkvb')]
    size = rest
    qkvb_suffix = '_qkvb' if qkvb else ''

    # timm naming convention used by DINOv3 ViTs
    # examples:
    #   vit_small_patch16_dinov3.lvd1689m
    #   vit_small_patch16_dinov3_qkvb.lvd1689m
    timm_model_name = f'vit_{size}_patch16_dinov3{qkvb_suffix}.lvd1689m'

    n_heads_per_size = {
        'tiny': 3,
        'small': 6,
        'small_plus': 6,
        'base': 12,
        'large': 16,
        'huge_plus': 20,
    }
    assert size in n_heads_per_size, f"Unknown DINOv3 size: '{size}'"

    # NOTE: create_model prunes all kwargs with value None so we need to set
    #       num_heads explicitly here to the intended default.
    n_heads = kwargs.get('num_heads', None)
    kwargs['num_heads'] = n_heads or n_heads_per_size[size]

    model = _create_model_with_eva_patch_embed_override(
        timm_model_name, pretrained, n_input_channels, **kwargs
    )
    return DinoV3Backbone(model=model, n_input_channels=n_input_channels)
