# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

from ...types import BatchType
from .base import PreprocessingBase
from .clone import clone_entries
from .base import MULTI_DOWNSCALE_KEY_FMT
from .resize import resize
from .utils import _keys_available
from .utils import _get_input_shape


def get_downscale(sample: BatchType, downscale: int) -> BatchType:
    return sample.get(MULTI_DOWNSCALE_KEY_FMT.format(downscale), None)


class MultiscaleSupervisionGenerator(PreprocessingBase):
    def __init__(
        self,
        downscales: Tuple[int],
        keys: Tuple[str]
    ) -> None:
        self._downscales = downscales
        self._keys = keys

        super().__init__(
            fixed_parameters={
                'downscales': self._downscales,
                'keys': self._keys
            },
            multiscale_processing=False  # it is creating the multiscale entries
        )

    @property
    def downscales(self):
        return self._downscales

    def _preprocess(
        self,
        sample: BatchType,
        **kwargs
    ) -> Tuple[BatchType, Dict[str, Any]]:
        if not _keys_available(sample, self._keys):
            raise KeyError(f"At least one key of '{self._keys}' is missing in"
                           "`sample`.")

        # read shape from rgb or depth image (at least one is available)
        h, w = _get_input_shape(sample)

        shape_dict = {}
        for downscale in self._downscales:
            # clone selected entries
            cloned_sample = clone_entries(sample, keys_to_clone=self._keys)

            # resize
            h_down, w_down = int(h / downscale), int(w / downscale)
            cloned_sample = resize(cloned_sample, height=h_down, width=w_down)
            shape_dict[downscale] = (h_down, w_down)

            # add resized multiscale clone to dict
            key = MULTI_DOWNSCALE_KEY_FMT.format(downscale)
            sample[key] = cloned_sample

        return sample, {'shapes': shape_dict}
