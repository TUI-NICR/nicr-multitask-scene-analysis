# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Iterable, Optional, Tuple
from copy import deepcopy

from ...types import BatchType
from .base import PreprocessingBase


DEFAULT_CLONE_KEY = '_no_preprocessing'


def clone_entries(
    sample: BatchType,
    keys_to_clone: Iterable[str],
    ignore_missing_keys: bool = False
) -> BatchType:
    return {
        key: deepcopy(sample[key])
        for key in keys_to_clone
        if key in sample or not ignore_missing_keys
    }


class CloneEntries(PreprocessingBase):
    def __init__(
        self,
        keys_to_clone: Optional[Iterable[str]] = None,
        ignore_missing_keys: bool = False,
        clone_key: str = DEFAULT_CLONE_KEY
    ) -> None:
        self._keys_to_clone = keys_to_clone
        self._ignore_missing_keys = ignore_missing_keys
        self._clone_key = clone_key

        super().__init__(
            fixed_parameters={
                'clone_key': self._clone_key,
                'ignore_missing_keys': self._ignore_missing_keys,
            },
            multiscale_processing=False
        )

    @property
    def clone_key(self):
        return self._clone_key

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        # copy all entries such that they are available for later
        keys = self._keys_to_clone or tuple(sample.keys())
        sample[self._clone_key] = clone_entries(sample, keys,
                                                self._ignore_missing_keys)

        return sample, {'cloned_keys': keys}


class FlatCloneEntries(PreprocessingBase):
    def __init__(
        self,
        keys_to_clone: Optional[Iterable[str]] = None,
        ignore_missing_keys: bool = False,
        key_prefix: Optional[str] = None,
        key_suffix: Optional[str] = None
    ) -> None:
        assert key_prefix or key_suffix

        self._keys_to_clone = keys_to_clone
        self._ignore_missing_keys = ignore_missing_keys
        self._key_prefix = key_prefix or ''
        self._key_suffix = key_suffix or ''

        super().__init__(
            fixed_parameters={
                'key_prefix': self._key_prefix,
                'key_suffix': self._key_suffix,
                'ignore_missing_keys': self._ignore_missing_keys,
            },
            multiscale_processing=False
        )

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        # clone entries
        keys = self._keys_to_clone or tuple(sample.keys())
        cloned_entries = clone_entries(sample, keys,
                                       self._ignore_missing_keys)

        # add entries with modified key to dict
        added_keys = []
        for k, v in cloned_entries.items():
            new_key = f'{self._key_prefix}{k}{self._key_suffix}'
            sample[new_key] = v
            added_keys.append(new_key)

        return sample, {'added_keys': added_keys}
