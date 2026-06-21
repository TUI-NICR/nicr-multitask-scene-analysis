# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, List, Sequence, Tuple

import torch

from ..base import TaskHelperBase


class TokenMatchingCache:
    def __init__(self):
        # _results[batch_idx][stage_key] stores the matcher output for one
        # decoder output stage. stage_key is 'main' or a side-output key.
        # The value is a list with one (pred_indices, target_indices) tuple
        # for each sample in the batch.
        self._results: Dict[
            int, Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = {}

    def __contains__(self, batch_idx: int) -> bool:
        return batch_idx in self._results

    def __getitem__(
        self,
        batch_idx: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.get_stage(batch_idx, 'main')

    def set(self, batch_idx: int,
            indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        indices_list = list(indices)
        self._results[batch_idx] = {'main': indices_list}

    def set_stage(
        self,
        batch_idx: int,
        stage_key: str,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        indices_list = list(indices)
        stage_map = self._results.setdefault(batch_idx, {})
        stage_map[stage_key] = indices_list

    def get_stage(
        self,
        batch_idx: int,
        stage_key: str = 'main'
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        stage_map = self._results.get(batch_idx)
        assert stage_map is not None, (
            'Token matching information is missing for batch '
            f'{batch_idx}. Run the token mask helper first.'
        )
        indices = stage_map.get(stage_key)
        assert indices is not None, (
            f"No matching stored for stage '{stage_key}' (batch {batch_idx})."
        )
        return indices

    def clear(self, batch_idx: int) -> None:
        self._results.pop(batch_idx, None)


class TokenTaskHelperBase(TaskHelperBase):
    # Shared base for token helpers. Not every token task uses query-target
    # matching, so matching support lives in TokenMatchingTaskHelperBase.
    pass


class TokenMatchingTaskHelperBase(TokenTaskHelperBase):
    def __init__(self, *, matching_cache: TokenMatchingCache):
        super().__init__()
        self._matching_cache = matching_cache

    @property
    def matching_cache(self) -> TokenMatchingCache:
        return self._matching_cache
