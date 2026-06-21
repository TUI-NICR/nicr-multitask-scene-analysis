# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from typing import Any, Dict, Tuple, Union

from torch import Tensor


_TensorTuple = Tuple[Tensor, ...]

# input
BatchType = Dict[str, Any]

# encoder
EncoderForwardType = Dict[str, Tensor]    # modality: features
EncoderTokenForwardType = Dict[str, Tensor]    # modality: extra tokens
EncoderInputType = EncoderForwardType
EncoderTokenInputType = EncoderTokenForwardType
EncoderSkipType = EncoderForwardType
EncoderSkipsType = Dict[str, EncoderSkipType]    # downsampling: skip features
EncoderOutputTypePlain = Tuple[EncoderForwardType, EncoderSkipsType]
EncoderOutputTypeExtraTokens = Tuple[EncoderForwardType,
                                     EncoderSkipsType,
                                     EncoderTokenForwardType]
EncoderOutputType = Union[EncoderOutputTypePlain, EncoderOutputTypeExtraTokens]

# context module
ContextModuleInputType = Tensor
ContextModuleContextFeaturesType = Union[_TensorTuple, Tuple[()]]
ContextModuleOutputType = Tuple[Tensor, ContextModuleContextFeaturesType]

# decoder
DecoderInputType = ContextModuleOutputType
DenseDecoderModuleSideOutputType = Union[Tensor, _TensorTuple, None]
DenseDecoderModuleOutputType = Tuple[Tensor,
                                     DenseDecoderModuleSideOutputType]
DecoderRawOutputType = Tuple[
    # outputs
    Union[
        Tensor,    # single tensor (semantic, scene, normal)
        _TensorTuple,    # multiple tensors (instance)
        Tuple[Tensor, _TensorTuple]    # panoptic (=semantic+instance)
    ],
    # side outputs
    Union[
        Tuple[DenseDecoderModuleSideOutputType, ...],    # semantic, instance, normal
        None,    # scene
        Tuple[Tuple[DenseDecoderModuleSideOutputType, ...],    # panoptic
              Tuple[DenseDecoderModuleSideOutputType, ...]]
    ]
]
DecoderPostprocessedOutputType = Dict[str, Any]
DecoderOutputType = Union[DecoderRawOutputType, DecoderPostprocessedOutputType]

TokenDecoderInputType = EncoderOutputTypeExtraTokens

# postprocessing
PostprocessingOutputType = DecoderPostprocessedOutputType
