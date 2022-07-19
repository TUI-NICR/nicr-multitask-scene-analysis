# NICR Multi-Task Scene Analysis
This repository provides some core functionality for multi-task scene analysis in PyTorch.
It contains essential functions for:
- panoptic segmentation (semantic + instance segmentation)
- instance orientation estimation
- scene classification

The repository builds upon the [NICR Scene Analysis Datasets](https://github.com/TUI-NICR/nicr-scene-analysis-datasets) repository.
In addition to that, it is used in our projects, e.g., [EMSANet](https://github.com/TUI-NICR/EMSANet).

> Note that this package is used in ongoing research projects and will be extended and maintained as needed.

## License and Citations
The source code is published under Apache 2.0 license, see [license file](LICENSE) for details.

If you use the source code, please cite the papers related to this work:

**Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments** ([arXiv](https://arxiv.org/abs/2207.04526)):
>Seichter, D., Fischedick, B., Köhler, M., Gross, H.-M.
*Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments*,
to appear in IEEE International Joint Conference on Neural Networks (IJCNN), 2022.

```bibtex
@article{emsanet2022,
title={Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments},
author={Seichter, Daniel and Fischedick, S{\"o}hnke and K{\"o}hler, Mona and Gross, Horst-Michael},
journal={arXiv preprint arXiv:2207.04526},
year={2022}
}
```
Note that the preprint was accepted to be published in IEEE International Joint Conference on Neural Networks (IJCNN) 2022.

## Installation
```bash
# option 1: directly install to your site packages
pip install git+https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git [--user]

# option 2: install editable version
git clone https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git
cd /path/to/this/repository

pip install -e . [--user]
```

Note, if you use this repository along with our other projects, please follow the installation instructions given there. This ensures installing the correct version.

After installation, the package can be imported by using the following command (note that the name is shortened):
```python
import nicr_mt_scene_analysis
```

We refer to the code and the short overview below for further details on the usage.

## Content
This repository provides some core functionality for multi-task scene analysis.
In the following section, some major components are listed ordered by their folder structure in the repository.

### Preprocessing
For preparing network inputs, different preprocessing and augmentation steps are required. Similar to [NICR Scene Analysis Datasets](https://github.com/TUI-NICR/nicr-scene-analysis-datasets), all preprocessors work inplace on a dict of inputs. Except of `TorchTransformWrapper` all processing steps are implemented using NumPy (not PyTorch).  
This package implements the following preprocessing modules:
- `CloneEntries`: Clones specific keys to an extra field in the input dict. This can be helpful if data should be kept without preprocessing.
- `FlatCloneEntries`: Similar to CloneEntires but appends a suffix to the keys and does not build a nested dict.
- `FullResCloner`: Similar to FlatCloneEntires and can be used to keep the input data in full resolution before resizing happens (use this preprocessor to ensure that validation is done on full resolution).
- `InstanceClearStuffIDs`: Enforces that all stuff pixels have the instance id equal to 0 (indicates *no_instance*).
- `InstanceTargetGenerator`: Uses the ground-truth instance image for generating instance offsets and center images (bottom-up instance segmentation, see [PanopticDeepLab](https://arxiv.org/abs/1911.10194)). This is required for training to compute losses.
- `MultiscaleSupervisionGenerator`: Generates copies of the ground truth in multiple resolutions. This is useful for training with multi-scale supervision.
- `NormalizeDepth`: Normalizes the depth image with a given depth mean and std.
- `NormalizeRGB`: Normalizes the RGB image with an ImageNet mean and std.
- `OrientationTargetGenerator`: Generates a dense orientation image that is required for loss computation.
- `PanopticTargetGenerator`: Combines both semantic and instance segmentation and converts it to a panoptic segmentation encoding.
- `RandomCrop`: Randomly crops the input to a specific width and height. This can be used for augmentation of the training data. Note, the same cropping is automatically applied to all spatial keys in the dict.
- `RandomHSVJitter`: Randomly adds a color jitter to the input RGB image in HSV space.
- `RandomHorizontalFlip`: Flips the image randomly at the horizontal axis for augmentation of the training data. Note, the same flipping is automatically applied to all spatial keys and given orientations in the dict.
- `RandomResize`: Randomly resizes for augmentation of the training data.  Note, the resizing is automatically applied to all spatial keys in the dict.
- `Resize`: Resize so the inputs fits a given size. Note, the same resizing is automatically applied to all spatial keys in the dict.
- `ToTorchTensors`: Converts the NumPy arrays to torch tensors.
- `TorchTransformWrapper`: Wrapper that enables using torchvision transforms with multi-modal input (after converting to torch tensors).


### Loss
The different tasks require different loss computations. The following classes are provided:
- `CrossEntropyLossSemantic`: Computes the cross entropy loss for the semantic segmentation.
- `L1Loss`: Computes the L1 loss that can be used, e.g., for instance offset loss computation.
- `MSELoss`: Computes the MSE loss that can be used, e.g., for instance center loss computation.
- `VonMisesLossBiternion`: Computes a dense version of the [VonMisesLoss](https://link.springer.com/chapter/10.1007/978-3-319-24947-6_13) in biternion encoding. This can be used for computing the loss for instance orientation estimation.

### Loss Weighting
In multi-task settings, some strategy is required for combining the losses of the different tasks.
- `DynamicWeightAverage`: Loss weighting according to [DWA](https://arxiv.org/pdf/1803.10704.pdf).
- `FixedLossWeighting`: Combines the losses by weighting each loss with a fixed weight and summing them up.
- `RandomLossWeighting`: Loss weighting according to [RandomLossWeighting](https://arxiv.org/abs/2111.10603).

### Metric
For evaluating different tasks in a multi-task setting, each tasks further requires a metric.
- `MeanAbsoluteAngularError`: Computes the MAAE for evaluating instance orientation estimation.
- `MeanIntersectionOverUnion`: Computes the mIoU for semantic segmentation.
- `PanopticQuality`: Computes the panoptic quality (PQ), segmentation quality (SQ), and recognition quality (RQ) according to the [Panoptic Segmentation](https://arxiv.org/abs/1801.00868).
- `PanopticQualityWithOrientationMAE`: Computes the PQ/SQ/RQ exactly like `PanopticQuality`. However, the provided matching during PQ computation is further used for calculating the MeanAbsoluteAngularError for evaluating instance orientation estimation.
- `RootMeanSquaredError`: Computes the RMSE for normal estimation.

### Model
Holds all network related code. Except the postprocessors, all components can be exported to ONNX.

#### Backbone
- `ResNetBackbone`: Implements the ResNet v1.5 (downsampling in 3x3 conv in bottleneck block) backbone.
- `ResNetSEBackbone`: Same as `ResNetBackbone` but with channel-wise squeeze-and-excitation (SE) at the end of each stage.

#### Block
- `BasicBlock`: Basic block of ResNet v1.
- `Bottleneck`: Bottleneck block of ResNet v1.5 (downsampling in 3x3 conv).
- `NonBottleneck1D`: similar to `BasicBlock` but with factorized convolutions as in [ERFNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf). This means a 3x3 conv is factorized into a 3x1 and a 1x3 conv with an activation function (typically ReLU) in between.

#### Context Module
- `AdaptivePyramidPoolingModule`: Pyramid Pooling Module of [PSPNet](https://arxiv.org/abs/1612.01105) with fixed output sizes (adaptive pooling sizes).
- `PyramidPoolingModule`: Pyramid Pooling Module of [PSPNet](https://arxiv.org/abs/1612.01105) with fixed pooling sizes (adaptive output size depending on the input).

#### Decoder
- `InstanceDecoder`: Convolution-based decoder for instance segmentation and instance orientation estimation with multi-scale output heads.
- `NormalDecoder`: Convolution-based decoder for normal estimation with multi-scale output heads.
- `PanopticHelper`: Wrapper that encapsulates an instance and a semantic decoder to enable combing both later in the postprocessing to a panoptic segmentation.
- `SceneClassificationDecoder`: MLP-based decoder (single layer) for scene classification.
- `SemanticDecoder`: Convolution-based decoder for semantic segmentation with multi-scale output heads.

#### Encoder
- `FusedEncoders`: Wrapper that combines up to two backbones (RGB/depth/both) in order to create a (multi-modality) (fused) encoder. It also handles creating all skip connections from encoder to the task-specific decoders. Moreover, if both modalities are present, it also handles fusing both at certain stages.

#### Encoder-Decoder-Fusion
- `EncoderDecoderFusionAdd`: Simple skip connection from encoder to decoder. Features from the encoder (after performing the encoder fusion) are added to decoder features of the same resolution in the decoder. Note if the number of channels in the decoder is different, an additional 1x1 conv is added to adjust the number of channels of the encoder features.

#### Encoder-Fusion
- `EncoderFusionWeightedAdd`: Simple channel-wise squeeze-and-excitation weighted unidirectional or bidirectional fusion between the both modalities in `FusedEncoders`.

#### Postprocessing
- `InstancePostprocessing`: Handles resizing and combining predicted instance centers and instance offsets to an instance segmentation based on a given (ground-truth) foreground mask. Moreover, if the orientation estimation task is present, it also handles deriving instance orientations for given (ground-truth) instances as well as the predicted instances. Note that the aforementioned postprocessing is only computed during validation in the training process. For inference, the foreground mask derived from semantic segmentation is used (see `PanopticPostprocessing`).
- `NormalPostprocessing`: Handles resizing the raw outputs of the normal estimation decoder.
- `PanopticPostprocessing`: Wrapper that encapsulates both an instance and a semantic postprocessor to enable panoptic segmentation. It first calls the encapsulated postprocessors and, subsequently, derives the panoptic segmentation based on the predicted semantic segmentation, i.e., the predicted semantic class decides whether a given pixel is considered as foreground (thing) or background (stuff). Moreover, if the orientation estimation task is present, it also handles deriving instance orientations based on the predicted panoptic instances.
- `ScenePostprocessing`: Handles postprocessing the raw outputs of the scene classification decoder, i.e., applies softmax and determines the argmax.
- `SemanticPostprocessing`: Handles resizing and postprocessing the raw outputs of the semantic decoder, i.e., applies softmax and determines the argmax.

#### Upsampling
- `UpsamplingX2`: Implements nearest and bilinear upsampling as well as our proposed learned upsampling. Note that the upsampling is always done by a factor of 2. Use multiple modules for larger factors.

### Task Helper
Task helpers handle the pipelines for both training and validation for each task. This helps keeping network related code separated from pipeline code and, thus, facilitate ONNX export.
Each task helper implements a `training_step` and a `validation_step`.
In training, only the loss gets computed. In validation, besides the loss, metrics are computed task.
- `InstanceTaskHelper`: Task helper for instance segmentation and instance orientation estimation. It computes the MSE/L1/VonMises loss for instance center/instance offset/instance orientation. In validation, it computes PQ/RQ/SQ to evaluate instance segmentation (using the ground-truth instance foreground) and MAAE to evaluate instance orientation estimation (using ground-truth instance orientation foreground).
- `NormalTaskHelper`: Task helper for normal estimation. Computes the L1 loss in training and RMSE metric in validation.
- `PanopticTaskHelper`: Calculates the PQ/RQ/SQ for the merged panoptic segmentation. Furthermore, it computes MAAE with predicted instances and mIoU after merging semantic and instance segmentation.
Note, no additional loss is calculated.
- `SceneTaskHelper`: Task helper for scene classification. Computes the cross entropy loss and the (balanced) accuracy Acc/bAcc.
- `SemanticTaskHelper`: Task helper for semantic segmentation. Computes the cross entropy loss and the mIoU.

### Visualization
Functions for visualizing the ground truth and prediction for each task.

### Other stuff
Some other stuff that might be useful to you.

- `CheckpointHelper`: Helps matching given short metric names (e.g., 'miou' or 'bacc') to actual metric key names of the task helpers. Furthermore, it tracks all matched metrics in order to determine whether a new best value was reached and, thus, a checkpoint should be created.
- `CSVLogger`: Simple metrics to CSV logger that is capable of handling changing keys.
