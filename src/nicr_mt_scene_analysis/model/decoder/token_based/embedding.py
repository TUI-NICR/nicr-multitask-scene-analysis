# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Type

from torch import nn

from ....types import DecoderRawOutputType
from ....types import EncoderSkipsType
from ....types import TokenDecoderInputType
from ...postprocessing import get_postprocessing_class
from ...postprocessing import PostProcessingType
from .base import TokenLinearHeadDecoder


class TokenEmbeddingDecoder(TokenLinearHeadDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        embedding_dim: int,
        modality: Optional[str] = None,
        linear_probing_classes: Optional[int] = None,
        linear_probing_classes_per_dataset: Optional[Dict[str, int]] = None,
        mlp_hidden_dim: int = 1024,
        mlp_dropout: float = 0.0,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class(
            'token-visual-embedding'
        )
    ):
        super().__init__(embed_dim=embed_dim,
                         output_dim=embedding_dim,
                         modality=modality,
                         postprocessing=postprocessing)
        input_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )
        self._linear_probing_head: Optional[nn.Module] = None
        if linear_probing_classes_per_dataset is not None:
            assert all(
                n_classes > 0
                for n_classes in linear_probing_classes_per_dataset.values()
            ), 'linear probing heads must have at least one class.'
            self._linear_probing_head = nn.ModuleDict({
                ds_name: nn.Linear(embedding_dim, n_classes)
                for ds_name, n_classes
                in linear_probing_classes_per_dataset.items()
            })
        elif linear_probing_classes is not None:
            self._linear_probing_head = nn.Linear(
                embedding_dim,
                linear_probing_classes
            )

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[Dict[str, Any]] = None
    ) -> DecoderRawOutputType:
        key, queries = self._resolve_queries(x)
        embeddings = self._compute_head_output(queries)
        side_outputs = self._collect_side_outputs(skips, key)
        if self._linear_probing_head is None:
            return embeddings, side_outputs
        detached = embeddings.detach()
        if isinstance(self._linear_probing_head, nn.ModuleDict):
            # multiple datasets: run the dataset-specific linear probe head.
            assert meta is not None, (
                "TokenEmbeddingDecoder requires meta for "
                "per-dataset linear probing."
            )
            max_classes = max(
                head.out_features for head in self._linear_probing_head.values()
            )
            # Stable softmax computes exp(logits - max(logits)). This is safe
            # for -inf padding as long as not all logits in a row are -inf:
            # each row gets finite logits from its dataset head, so max(logits)
            # is finite and padded slots become exp(-inf) = 0.
            fill_value = float('-inf')
            linear_logits = detached.new_full(
                (detached.shape[0], detached.shape[1], max_classes),
                fill_value
            )
            for sample_idx in range(detached.shape[0]):
                dataset_name = meta[sample_idx]['dataset_type'].__name__
                if dataset_name not in self._linear_probing_head:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in linear "
                        "probing head datasets: "
                        f"{list(self._linear_probing_head.keys())}"
                    )
                head = self._linear_probing_head[dataset_name]
                sample_logits = head(
                    detached[sample_idx:sample_idx+1]
                ).squeeze(0)
                linear_logits[
                    sample_idx, :, :sample_logits.shape[-1]
                ] = sample_logits
        else:
            # single dataset: one linear probe head is shared by all samples.
            linear_logits = self._linear_probing_head(detached)
        return (embeddings, linear_logits), side_outputs


class TokenImageEmbeddingDecoder(TokenLinearHeadDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        embedding_dim: int,
        modality: Optional[str] = None,
        n_scene_classes_per_dataset: Optional[Dict[str, int]] = None,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class(
            'token-image-embedding'
        )
    ):
        super().__init__(embed_dim=embed_dim,
                         output_dim=embedding_dim,
                         modality=modality,
                         postprocessing=postprocessing,
                         use_cls_token=True)

        self._linear_probing_head: Optional[nn.ModuleDict] = None
        if n_scene_classes_per_dataset is not None:
            self._linear_probing_head = nn.ModuleDict({
                ds_name: nn.Linear(embedding_dim, scene_classes)
                for ds_name, scene_classes
                in n_scene_classes_per_dataset.items()
            })

    def _forward_training(
        self,
        x: TokenDecoderInputType,
        skips: EncoderSkipsType,
        meta: Optional[Dict[str, Any]] = None
    ) -> DecoderRawOutputType:
        # scene classification operates on the CLS token only and does not need
        # side outputs.
        _, queries = self._resolve_queries(x)
        embeddings = self._compute_head_output(queries)

        scene_logits = []
        if self._linear_probing_head is not None:
            detached_embeddings = embeddings.detach()

            for sample_idx in range(detached_embeddings.shape[0]):
                dataset_name = meta[sample_idx]['dataset_type'].__name__
                if dataset_name not in self._linear_probing_head:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in linear "
                        "probing head datasets: "
                        f"{list(self._linear_probing_head.keys())}"
                    )
                linear_head = self._linear_probing_head[dataset_name]
                sample_embedding = detached_embeddings[
                    sample_idx:sample_idx+1,
                    :
                ]
                sample_scene_logits = linear_head(sample_embedding)
                sample_scene_logits = sample_scene_logits.squeeze(1)
                scene_logits.append(sample_scene_logits)

        return (embeddings, scene_logits), None
