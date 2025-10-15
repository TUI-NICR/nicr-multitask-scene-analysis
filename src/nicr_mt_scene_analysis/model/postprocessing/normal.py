# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from ...data.preprocessing.resize import get_fullres_key
from ...data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .dense_base import DensePostprocessingBase


class NormalPostprocessing(DensePostprocessingBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {
            'normal_output': output,
            'normal_side_outputs': side_outputs
        }

        return r_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs (there are no side outputs)
        output, side_outputs = data

        # create results dict
        r_dict = {
            'normal_output': output,
            'normal_side_outputs': side_outputs
        }

        # crop and resize output to full resolution (original shape)
        # note, we assume same shape for all samples in batch
        crop_slices, resize_shape = get_valid_region_slices_and_fullres_shape(
            batch, 'normal'
        )

        output_fullres = self._crop_to_valid_region_and_resize_prediction(
            output, valid_region_slices=crop_slices, shape=resize_shape,
            mode='nearest'
        )

        # update results dict
        r_dict.update({
            get_fullres_key('normal_output'): output_fullres
        })

        return r_dict
