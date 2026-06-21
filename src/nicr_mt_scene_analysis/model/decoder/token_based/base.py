# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import Optional, Sequence, Tuple, Type

from torch import Tensor
import torch.nn as nn

from ....types import BatchType
from ....types import DecoderOutputType
from ....types import DecoderRawOutputType
from ....types import EncoderForwardType
from ....types import EncoderSkipsType
from ....types import EncoderTokenForwardType
from ....types import PostprocessingOutputType
from ....types import TokenDecoderInputType
from ...encoder import ENCODER_META_KEY
from ...postprocessing import PostProcessingType
from ...postprocessing.token_based.base import TokenPostprocessingBase
from ..base import DecoderBase


class TokenDecoderBase(DecoderBase):
    def __init__(
        self,
        *,
        embed_dim: int,
        modality: Optional[str] = None,
        postprocessing: Type[PostProcessingType],
        side_output_stage_indices: Optional[Sequence[int]] = None
    ):
        super().__init__(postprocessing=postprocessing)

        self._embed_dim = embed_dim
        self._modality = modality
        self._side_stage_indices = (
            tuple(side_output_stage_indices)
            if side_output_stage_indices is not None
            else None
        )

    def forward(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        batch: BatchType,
        do_postprocessing: bool = True,
        out_dict: Optional[PostprocessingOutputType] = None
    ) -> DecoderOutputType:
        input_meta = batch.get('meta', None)

        if self.training:
            output = self._forward_training(x, skips, input_meta)
        else:
            output = self._forward_inference(x, skips, input_meta)

        if do_postprocessing:
            if out_dict is None:
                out_dict = {}
            # token postprocessors read prior decoders' outputs from the shared
            # out_dict and merge into it. dense ones (e.g. ScenePostprocessing)
            # keep the base signature and return a fresh dict. merging the
            # result covers both cases. merging out_dict into itself is a no-op.
            kwargs = {'is_training': self.training}
            if isinstance(self._postprocessing, TokenPostprocessingBase):
                kwargs['out_dict'] = out_dict
            postprocessed = self._postprocessing.postprocess(
                output, batch, **kwargs
            )
            out_dict.update(postprocessed)
            return out_dict

        return output

    @abc.abstractmethod
    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> DecoderRawOutputType:
        pass

    def _forward_inference(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> DecoderRawOutputType:
        # token-aware default. concrete decoders may override with custom
        # inference logic but must keep the meta kwarg.
        return self._forward_training(x, skips, meta)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def modality(self) -> Optional[str]:
        return self._modality

    def _available_modalities(self, x: EncoderForwardType) -> Tuple[str, ...]:
        return tuple(k for k in x.keys() if k != ENCODER_META_KEY)

    def _select_modality(self, x: EncoderForwardType) -> str:
        modalities = self._available_modalities(x)
        assert modalities, 'Token decoders require encoder features.'
        if self.modality is not None:
            if self.modality not in modalities:
                raise ValueError(
                    f"Modality '{self.modality}' not available. "
                    f"Available: {modalities}"
                )
            return self.modality
        if len(modalities) != 1:
            raise ValueError(
                'modality must be specified when multiple encoder outputs '
                'exist.'
            )
        # only one modality available
        return modalities[0]

    @staticmethod
    def _collect_queries(
        tokens_dict: EncoderTokenForwardType,
        key: str
    ) -> Tensor:
        tensor = tokens_dict.get(key)
        if tensor is None:
            raise ValueError(
                f"No query/extra tokens available for modality '{key}'."
            )
        assert tensor.dim() == 3
        return tensor

    def _iter_side_entries(
        self,
        skips: EncoderSkipsType
    ) -> Tuple[dict, ...]:
        entries = tuple(skips.values())
        if not entries:
            return ()
        if self._side_stage_indices is None:
            return entries
        stage_map = {
            entry.get('_stage_idx'): entry
            for entry in entries
            if isinstance(entry, dict) and '_stage_idx' in entry
        }
        return tuple(
            stage_map[idx]
            for idx in self._side_stage_indices
            if idx in stage_map
        )


class TokenLinearHeadDecoder(TokenDecoderBase):
    def __init__(
        self,
        *,
        embed_dim: int,
        output_dim: int,
        modality: Optional[str] = None,
        squeeze_queries: bool = False,
        output_transform: Optional[nn.Module] = None,
        postprocessing: Type[PostProcessingType],
        side_output_stage_indices: Optional[Sequence[int]] = None,
        use_cls_token: bool = False
    ):
        super().__init__(embed_dim=embed_dim,
                         modality=modality,
                         postprocessing=postprocessing,
                         side_output_stage_indices=side_output_stage_indices)
        self.head = nn.Linear(embed_dim, output_dim)
        self._squeeze_queries = squeeze_queries
        self._output_transform = output_transform or nn.Identity()
        # reuse the backbone's CLS token instead of caller-injected extras
        self._use_cls_token = use_cls_token

    def _validate_queries(self, output: Tensor) -> None:
        assert not self._squeeze_queries or output.shape[1] == 1, \
            'TokenLinearHeadDecoder expects exactly one query when ' \
            'squeeze_queries=True.'

    def _resolve_queries(
        self,
        x: TokenDecoderInputType
    ) -> Tuple[str, Tensor]:
        tokens_by_modality, _, extra_tokens = x
        key = self._select_modality(tokens_by_modality)
        if self._use_cls_token:
            return key, tokens_by_modality[key][:, :1, :]
        queries = self._collect_queries(extra_tokens, key)
        return key, queries

    def _compute_head_output(self, queries: Tensor) -> Tensor:
        output = self.head(queries)
        output = self._output_transform(output)
        self._validate_queries(output)
        if self._squeeze_queries:
            output = output[:, 0, :]
        return output

    def _collect_side_outputs(
        self,
        skips: EncoderSkipsType,
        key: str
    ) -> Tuple[Tensor, ...]:
        side_outputs = []
        for entry in self._iter_side_entries(skips):
            tokens_dict = entry.get('_extra_tokens')
            if tokens_dict is None or key not in tokens_dict:
                continue
            queries_aux = self._collect_queries(tokens_dict, key)
            side_outputs.append(self._compute_head_output(queries_aux))
        return tuple(side_outputs)

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[dict] = None
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        key, queries = self._resolve_queries(x)
        logits = self._compute_head_output(queries)
        side_outputs = self._collect_side_outputs(skips, key)
        return logits, side_outputs
