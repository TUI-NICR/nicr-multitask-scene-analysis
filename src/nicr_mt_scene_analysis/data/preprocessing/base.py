# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple

import abc
import re

from ...types import BatchType
from .._types import AppliedPreprocessingMeta
from .._types import PreprocessingParameterDict


MULTI_DOWNSCALE_KEY_FMT = '_down_{}'
APPLIED_PREPROCESSING_KEY = '_applied_preprocessing'


def get_applied_preprocessing_meta(
    sample: BatchType
) -> AppliedPreprocessingMeta:
    if APPLIED_PREPROCESSING_KEY not in sample:
        sample[APPLIED_PREPROCESSING_KEY] = AppliedPreprocessingMeta()

    return sample[APPLIED_PREPROCESSING_KEY]


def add_to_applied_preprocessing_meta(
    sample: BatchType,
    **parameters: Dict[str, Any]
) -> BatchType:
    # get or add applied preprocessing meta information to sample
    applied_preprocessing = get_applied_preprocessing_meta(sample)
    # applied preprocessing is mutable, so we can directly update it
    applied_preprocessing.append(PreprocessingParameterDict(**parameters))

    return sample


class PreprocessingBase(abc.ABC):
    def __init__(
        self,
        fixed_parameters: Optional[Dict[str, Any]] = None,
        multiscale_processing: bool = False
    ) -> None:
        self._multiscale_processing = multiscale_processing

        self._fixed_parameters = {
            'type': self.__class__.__name__,
            'multiscale_processing': self._multiscale_processing
        }

        if fixed_parameters is not None:
            self._fixed_parameters.update(fixed_parameters)

    @property
    def fixed_parameters(self) -> Dict[str, Any]:
        return self._fixed_parameters

    @abc.abstractmethod
    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        pass

    def __repr__(self) -> str:
        parameter_str = ', '.join(
            f"{k}: {v}" for k, v in self.fixed_parameters.items()
        )
        return f"{self.__class__.__name__}({parameter_str})"

    def __call__(self, sample: BatchType, **kwargs) -> BatchType:
        # preprocess sample
        sample, dynamic_parameters = self._preprocess(sample, **kwargs)

        # if multiscale processing is enabled, apply preprocessing to
        # multiscale entries as well
        multiscale_dynamic_parameters = {}
        if self._multiscale_processing:
            for key in sample:
                res = re.match(MULTI_DOWNSCALE_KEY_FMT.format('([0-9]+)'), key)
                if res is None:
                    # key does not match multiscale key format -> skip
                    continue

                # apply preprocessing to multiscale entry
                sample[key], ds_dynamic_parameters = self._preprocess(
                    sample[key], downscale=int(res.groups()[0]), **kwargs
                )
                multiscale_dynamic_parameters[key] = ds_dynamic_parameters

        # add/update applied preprocessing meta information in sample
        sample = add_to_applied_preprocessing_meta(
            sample,
            **self.fixed_parameters,
            **dynamic_parameters,
            **multiscale_dynamic_parameters
        )

        return sample
