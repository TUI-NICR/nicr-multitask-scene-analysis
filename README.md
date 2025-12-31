# NICR Multi-Task Scene Analysis
This repository provides some core functionality for multi-task scene analysis in PyTorch.
It contains essential functions for:

- panoptic segmentation (semantic + instance segmentation)
- instance orientation estimation
- scene classification
- dense visual embedding prediction (i.e. text aligned pixelwise embeddings)

with ResNet / Swin Transformer based encoder-decoder architectures processing RGB / Depth / RGB-D inputs.

The repository builds upon the [NICR Scene Analysis Datasets](https://github.com/TUI-NICR/nicr-scene-analysis-datasets) repository and is used in our projects:
- [EMSANet](https://github.com/TUI-NICR/EMSANet)
- [EMSAFormer](https://github.com/TUI-NICR/EMSAFormer)
- [Panoptic Mapping](https://github.com/TUI-NICR/panoptic-mapping)
- [DVEFormer](https://github.com/TUI-NICR/DVEFormer)

> Note that this package is used in ongoing research projects and will be extended and maintained as needed. Backward compatibility might be broken in new versions.

## License and Citations
The source code is published under Apache 2.0 license, see [license file](LICENSE) for details.

If you use the source code, please cite the paper related to your work:
---
**Efficient Prediction of Dense Visual Embeddings via Distillation and RGB-D Transformers** ([IEEE Xplore](https://ieeexplore.ieee.org/document/11245809)):
> Fischedick, S., Seichter, D., Stephan, B., Schmidt, R., Gross, H.-M.
*Efficient Prediction of Dense Visual Embeddings via Distillation and RGB-D Transformers*, in
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2400-2407, 2025.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{dveformer2025iros,  
  title     = {{Efficient Prediction of Dense Visual Embeddings via Distillation and RGB-D Transformers}},
  author    = {Fischedick, S{\"o}hnke and Seichter, Daniel and Stephan, Benedict and Schmidt, Robin and Gross, Horst-Michael},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {2400-2407},
  year      = {2025}
}
```
</details>

---

**PanopticNDT: Efficient and Robust Panoptic Mapping** ([IEEE Xplore](https://ieeexplore.ieee.org/document/10342137), [arXiv](https://arxiv.org/abs/2309.13635) (with appendix and some minor fixes)):
> Seichter, D., Stephan, B., Fischedick, S. B., Müller, S., Rabes, L., Gross, H.-M.
*PanopticNDT: Efficient and Robust Panoptic Mapping*,
in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{panopticndt2023iros,
  title     = {{PanopticNDT: Efficient and Robust Panoptic Mapping}},
  author    = {Seichter, Daniel and Stephan, Benedict and Fischedick, S{\"o}hnke Benedikt and  Mueller, Steffen and Rabes, Leonard and Gross, Horst-Michael},
  booktitle = {IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)},
  year      = {2023}
}
```

</details>

---

**Efficient Multi-Task Scene Analysis with RGB-D Transformers** ([IEEE Xplore](https://ieeexplore.ieee.org/document/10191977), [arXiv](https://arxiv.org/abs/2306.05242)):
> Fischedick, S., Seichter, D., Schmidt, R., Rabes, L., Gross, H.-M.
*Efficient Multi-Task Scene Analysis with RGB-D Transformers*,
in IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1-10, 2023.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{emsaformer2023ijcnn,
  title     = {{Efficient Multi-Task Scene Analysis with RGB-D Transformers}},
  author    = {Fischedick, S{\"o}hnke and Seichter, Daniel and Schmidt, Robin and Rabes, Leonard and Gross, Horst-Michael},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2023},
  pages     = {1-10},
  doi       = {10.1109/IJCNN54540.2023.10191977}
}
```

</details>

---

**Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9892852), [arXiv](https://arxiv.org/abs/2207.04526)):
> Seichter, D., Fischedick, S., Köhler, M., Gross, H.-M.
*Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments*,
in IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1-10, 2022.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{emsanet2022ijcnn,
  title     = {{Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments}},
  author    = {Seichter, Daniel and Fischedick, S{\"o}hnke and K{\"o}hler, Mona and Gross, Horst-Michael},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2022},
  pages     = {1-10},
  doi       = {10.1109/IJCNN55064.2022.9892852}
}
```

</details>

---

## Installation
To use our `nicr-multitask-scene-analysis` package, you must install OpenCV, PyTorch, and TorchVision first (see [PyTorch documentation](https://pytorch.org/get-started/locally/)).
The code was tested with PyTorch 1.10, 1.13, 2.0, 2.3 as well as 2.8.

```bash
# requirements:
# - PyTorch, TorchVision (see note above)
# - NICR Scene Analysis Datasets (see below)
# - all remaining dependencies are installed automatically
python -m pip install "git+https://github.com/TUI-NICR/nicr-scene-analysis-datasets.git@v0.8.3"

# option 1: directly install to your site packages
python -m pip install "git+https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git"

# option 2: install editable version
git clone https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git
cd /path/to/this/repository
python -m pip install -e "./"
```

If you want pip to also install the optional requirement sets we provide for PyTorch (`withtorch`) and OpenCV (`withopencv`), append the extras to the package specifier.
You can choose either extra individually or combine both:

```bash
# option 1: direct install in your site packages with extras
python -m pip install "nicr-mt-scene-analysis[withtorch,withopencv] @ git+https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git"

# option 2: editable install with extras
python -m pip install -e "./[withtorch,withopencv]"
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
- `DenseVisualEmbeddingTargetGenerator`: Uses panoptic segmentation generated by `PanopticTargetGenerator`  and embeddings provided along with the samples to generate an embedding lookup table and a dense index image. These can be combined to a dense visual embedding image e.g. for knowledge distillation.
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
- `ScaleDepth`: Similar to `NormalizeDepth` but scales the depth values sample-wise to a given range.
- `SemanticClassMapper`: Map semantic classes in a sample to a new label. This can be helpful for example for ScanNet to map semantic classes ignored in the benchmark to void.
- `ToTorchTensors`: Converts the NumPy arrays to torch tensors.
- `TorchTransformWrapper`: Wrapper that enables using torchvision transforms with multi-modal input (after converting to torch tensors).


### Loss
The different tasks require different loss computations. The following classes are provided:
- `CrossEntropyLossSemantic`: Computes the cross entropy loss for the semantic segmentation.
- `CosineEmbeddingLoss`: Computes the cosine distance loss for the dense visual embedding prediction.
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
For exporting Swin Transformer backbones, see: [EMSAFormer](https://github.com/TUI-NICR/EMSAFormer)

#### Backbone
- `ResNetBackbone`: Implements the ResNet v1.5 (downsampling in 3x3 conv in bottleneck block) backbone.
- `ResNetSEBackbone`: Same as `ResNetBackbone` but with channel-wise squeeze-and-excitation (SE) at the end of each stage.
- `SwinBackbone`: Implements the Swin Transformer and Swin Transformer v2 backbone.
- `SwinMultimodalBackbone`: Similar as `SwinBackbone` but with modifications to support multimodal (RGBD) input in a single backbone.

#### Block
- `BasicBlock`: Basic block of ResNet v1.
- `Bottleneck`: Bottleneck block of ResNet v1.5 (downsampling in 3x3 conv).
- `NonBottleneck1D`: similar to `BasicBlock` but with factorized convolutions as in [ERFNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf). This means a 3x3 conv is factorized into a 3x1 and a 1x3 conv with an activation function (typically ReLU) in between.

#### Context Module
- `AdaptivePyramidPoolingModule`: Pyramid Pooling Module of [PSPNet](https://arxiv.org/abs/1612.01105) with fixed output sizes (adaptive pooling sizes).
- `PyramidPoolingModule`: Pyramid Pooling Module of [PSPNet](https://arxiv.org/abs/1612.01105) with fixed pooling sizes (adaptive output size depending on the input).

#### Decoder
- `EmbeddingDecoder`: Convolution-based decoder for outputting a fixed-size
pixelwise embedding with multi-scale output heads.
- `EmbeddingMLPDecoder`: Same as `EmbeddingDecoder` but MLP-based (similar to SegFormer).
- `InstanceDecoder`: Convolution-based decoder for instance segmentation and instance orientation estimation with multi-scale output heads.
- `InstanceMLPDecoder`: Same as `InstanceDecoder` but MLP-based (similar to SegFormer).
- `NormalDecoder`: Convolution-based decoder for normal estimation with multi-scale output heads.
- `NormalMLPDecoder`: Same as `NormalDecoder` but MLP-based (similar to SegFormer).
- `PanopticHelper`: Wrapper that encapsulates an instance and a semantic decoder to enable combing both later in the postprocessing to a panoptic segmentation.
- `SceneClassificationDecoder`: MLP-based decoder (single layer) for scene classification.
- `SemanticDecoder`: Convolution-based decoder for semantic segmentation with multi-scale output heads.
- `SemanticMLPDecoder`: Same as `SemanticDecoder` but MLP-based (similar to SegFormer).

#### Encoder
- `Encoder`: Wrapper for a single backbone (RGB/depth/RGBD), which also handles creating all skip connections from encoder to the task-specific decoders.
- `FusedRGBDEncoder`: Similar wrapper as `Encoder` that combines two backbones (RGB and depth) in order to create a multi-modality fused encoder, which also handles fusing both modalities at certain stages.

#### Encoder-Decoder-Fusion
- `EncoderDecoderFusion`: A generic encoder-decoder fusion module that takes a fusion operation and source modality as arguments to fuse features from encoder to decoder. Note if the number of channels in the decoder is different, an additional 1x1 conv is added to adjust the number of channels of the encoder features.
- `EncoderDecoderFusionSwin`: Same as `EncoderDecoderFusion` but with an additional LayerNorm in the skip connection. It further ensures NCHW memory layout used in the decoders.

#### Encoder-Fusion
- `EncoderRGBDFusionWeightedAdd`: Simple channel-wise fusion between the two modalities in `FusedRGBDEncoder`. This can be done unidirectional or bidirectional and can be a simple addition or a channel-wise squeeze-and-excitation weighted addition.

#### Postprocessing
- `InstancePostprocessing`: Handles resizing and combining predicted instance centers and instance offsets to an instance segmentation based on a given (ground-truth) foreground mask. Moreover, if the orientation estimation task is present, it also handles deriving instance orientations for given (ground-truth) instances as well as the predicted instances. Note that the aforementioned postprocessing is only computed during validation in the training process. For inference, the foreground mask derived from semantic segmentation is used (see `PanopticPostprocessing`).
- `NormalPostprocessing`: Handles resizing the raw outputs of the normal estimation decoder.
- `PanopticPostprocessing`: Wrapper that encapsulates both an instance and a semantic postprocessor to enable panoptic segmentation. It first calls the encapsulated postprocessors and, subsequently, derives the panoptic segmentation based on the predicted semantic segmentation, i.e., the predicted semantic class decides whether a given pixel is considered as foreground (thing) or background (stuff). Moreover, if the orientation estimation task is present, it also handles deriving instance orientations based on the predicted panoptic instances.
- `ScenePostprocessing`: Handles postprocessing the raw outputs of the scene classification decoder, i.e., applies softmax and determines the argmax.
- `SemanticPostprocessing`: Handles resizing and postprocessing the raw outputs of the semantic decoder, i.e., applies softmax and determines the argmax.
- `DenseVisualEmbeddingPostprocessing`: Computes cosine similarity between the output embeddings and semantic text embeddings. The output is used to retrive semantic segmentation predictions with softmax and argmax.

#### Upsampling
- `Upsampling`: Implements nearest and bilinear upsampling as well as our proposed learned upsampling. Note that the learned upsampling is always done by a factor of 2. Use multiple modules for larger factors.

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
- `DenseVisualEmbeddingTaskHelper`: Task helper for dense visual embedding prediction. Computes the cosine embedding loss for knowledge distillation as well as the mIoU for text-based and visual-mean based semantic segmentation.

### Visualization
Functions for visualizing the ground truth and prediction for each task.

### Other stuff
Some other stuff that might be useful to you.

- `CheckpointHelper`: Helps matching given short metric names (e.g., 'miou' or 'bacc') to actual metric key names of the task helpers. Furthermore, it tracks all matched metrics in order to determine whether a new best value was reached and, thus, a checkpoint should be created.
- `CSVLogger`: Simple metrics to CSV logger that is capable of handling changing keys.


## Changelog

> Most relevant changes are listed below. Note that backward compatibility might be broken.

**Version 0.3.1 (Jan 04, 2026)**
- update DVEFormer citations


**Version 0.3.0 (Oct 15, 2025)**
- switch from from flat to src layout
- transition from setup.py to pyproject.toml for packaging (note that this means we are no longer able to detect
  whether opencv and/or torch are installed, instead this check is now done at runtime)
- use ruff for linting
- remove Python 3.6 at all, do not test for Python 3.11 anymore, add testing for Python 3.12
- refactor preprocessing:
  - add base class, store relevant preprocessing parameters in an additional entry in dict to retrace preprocessing steps
  - use `meta` dictionaries within preprocessing which enables to mix datasets with different thing/stuff class information
  - refactor multiscale supervision handling, see `multiscale_processing` argument in constructors
  - add `_get_relevant_tensor_keys` helper and enforce spatial dimensionality in `_get_relevant_spatial_keys`
  - add `keys_to_ignore` to `RandomResize` preprocessor
  - add `ScaleDepth` preprocessor
  - add `invalid_depth_value` to `NormalizeDepth` preprocessor
  - fix `TorchTransformWrapper` to work with torchvision.transforms.Compose containing a final FiveCrop/TenCrop
  - extend `Resize`/`RandomResize` to support `keep_aspect_ratio` workflows (e.g., ADE20K) and improved crop handling
  - adapt tests
- add `DenseVisualEmbeddingTargetGenerator` preprocessor for loading dense visual embeddings and align them with panoptic segmentation masks
- add `EmbeddingDecoder`and `EmbeddingMLPDecoder` for dense visual embedding prediction
- add `DenseVisualEmbeddingPostprocessing` for postprocessing of dense visual embeddings (e.g. use text embeddings for semantic segmentation prediction)
- add `DenseVisualEmbeddingTaskHelper` for dense visual embedding prediction training and validation
- add `CosineEmbeddingLoss` for computing the cosine distance loss for dense visual embedding prediction
- improve `move_batch_to_device` to traverse nested dict/list structures
- replaced some defaultdicts with normal dicts for torch.compile support
- fixed final file write of `CSVLogger`
- fix mIoU computation and test case for newer torchmetrics versions

**Version 0.2.3 (Jun 26, 2024)**
- add support for MPS device (only inference tested, training might work as well)
- add support for inference with CPU device
- fix inplace bug in `PanopticPostprocessing` (only when device is CPU)
- copy predicted orientations to instance meta dict (only for instances from panoptic segmentation)
- visualization:
  - add `cross_thickness` and `cross_markersize` to `visualize_instance_center*`
  - add `background_color` to `visualize_instance_offset*`
  - small fix in `InstanceColorGenerator` - first color of given colormap was not used (new visualizations will have a color shift)
  - use 'coolwarm' instead 'gray' colormap in `visualize_instance_center*` (this allows to better distinguish between zero values and masked areas)
- force 'panoptic' sample key to always be of dtype uint32 (see nicr-scene-analysis-datasets v0.6.1 release notes)
- ensure correct dtype handling in `resize` (dtype for panoptic was changed and wrongly handled)
- add support for resizing uint32 panoptic with OpenCV (fallback to 4*uint8, only for nearest-neighbor interpolation)
- add upcasts from uint16 to int32 and from uint32 to int64 in `ToTorchTensors`
- add `keys_to_ignore` to `Resize` preprocessor
- fix for PIL/Pillow 10
- do not use black (first color in default colormap) in `InstanceColorGenerator` for first instance id when no `cmap_without_void` is passed to constructor
- fix minor PIL issue (np.asarray / np.asanyarray return read-only arrays, np.array is used now)
- use pooling indices to disambiguate local maxima in instance center derivation (note, this might change metrics slightly; however, it resolves instance-center-assignment issues when using lower precisions, i.e., float16 or even more quantized types) -- thanks to Benedict Stephan for debugging and fixing this issue

**Version 0.2.2 (Sep 26, 2023)**
- add support for individual subset selection to `RandomSamplerSubset` when using with concatenated datasets (ConcatDataset) - requires `nicr-scene-analysis-datasets` >= 0.5.6
- add seed argument to `PanopticColorGenerator`
- tests: some fixes, skip testing with Python 3.6, add testing with Python 3.11

**Version 0.2.1 (Apr 20, 2023)**
- fix bug in `task_helper/instance.py`: metric object for mean absolute angular error was not reset after computing the metric (at the end of an epoch)

**Version 0.2.0 (Mar 28, 2023)**
- add Swin Transformers as backbone
- add ResNet with less downsampling
- add new dense MLP-based decoders for semantic, instance, and normal
- add more configurations with different bins to the context modules to support other input resolutions
- add SemanticClassMapper preprocessing module for mapping semantic classes to a new label
- change EncoderType and EncoderSkipType from tuple to dict
- extend mIoU to return IoUs per class for more detailed evaluation
- extend code to also support NHWC memory layout for Swin Transformer backbone
- extend and refactor postprocessing to return meta information such as semantic, instance, and panoptic scores
- refactor encoders for better single-encoder support
- refactor decoders to support dynamic encoder-decoder skip connections
- refactor visualization
- removed deprecated code
- extended test cases
