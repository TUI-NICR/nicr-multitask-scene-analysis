# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
from typing import Any, Iterator, List, Sized, Sequence, Tuple, Type, Union

import random

import numpy as np
from nicr_scene_analysis_datasets import ConcatDataset
import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate

from ._types import AppliedPreprocessingMeta
from ._types import CollateIgnoredDict


def collate(
    data: List[Any],
    type_blacklist: Tuple[Type] = (np.ndarray, ),
    default_type_blacklist: Tuple[Type] = (CollateIgnoredDict,
                                           AppliedPreprocessingMeta)
) -> Any:
    # input for first call is List[BatchType]

    # get first element for type and dict keys later
    elem = data[0]

    if isinstance(elem, type_blacklist+default_type_blacklist):
        # do not modify blacklisted types, e.g., numpy arrays, keep as list
        return data

    if isinstance(elem, torch.Tensor):
        # list of tensor, check shape
        if not all(t.shape == elem.shape for t in data):
            # we cannot stack the tensors as batch, keep a list of tensors
            return data

    if not isinstance(elem, dict):
        # use default pytorch collate
        return default_collate(data)

    # it is a dict
    return {
        key: collate([s[key] for s in data], type_blacklist)
        for key in elem
    }


class RandomSamplerSubset(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        subset: Union[float, Sequence[float]] = 1.0,
        deterministic: bool = False
    ) -> None:
        self._data_source = data_source
        self.subset = subset
        self.deterministic = deterministic

        if all((
            isinstance(self._data_source, ConcatDataset),
            isinstance(self.subset, (list, tuple))
        )):
            # dataset is a concatenated dataset, so subset should be given as
            # sequence of floats
            assert len(self.subset) == len(self._data_source.datasets)

    def __iter__(self) -> Iterator[int]:
        # determine seed
        if self.deterministic:
            # always the same indices on each call but in random order
            seed = 0
        else:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        # determine indices for subset
        if isinstance(self.subset, (list, tuple)):
            # we have a subset for each dataset in the concatenated dataset
            assert len(self.subset) == len(self._data_source.datasets)

            # get indices for each dataset
            indices = []
            start_idx = 0
            for ds, s in zip(self._data_source.datasets, self.subset):
                ds_indices = torch.randperm(len(ds), generator=generator)
                ds_indices += start_idx
                ds_indices = ds_indices.tolist()[:int(len(ds) * s)]
                start_idx += len(ds)

                indices.extend(ds_indices)
        else:
            # default, dataset is a single dataset
            indices = torch.randperm(len(self._data_source), generator=generator)
            indices = indices.tolist()[:len(self)]

        # shuffle the indices
        random.shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        if isinstance(self.subset, (list, tuple)):
            # we have a subset for each dataset in the concatenated dataset
            assert len(self.subset) == len(self._data_source.datasets)

            length = 0
            for ds, s in zip(self._data_source.datasets, self.subset):
                length += int(len(ds) * s)
            return length

        # default, dataset is a single dataset
        return int(len(self._data_source) * self.subset)
