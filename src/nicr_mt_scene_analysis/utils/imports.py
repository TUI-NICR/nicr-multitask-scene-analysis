# -*- coding: utf-8 -*-
"""
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional

import sys

from nicr_scene_analysis_datasets.utils.imports import DependencyImportHook
from nicr_scene_analysis_datasets.utils.imports import is_package_available


def is_opencv_available(
    raise_error: bool,
    min_version: Optional[str] = None
) -> bool:
    # might be NVIDIA's opencv, apt's python3-opencv, or opencv-python from PyPI
    return is_package_available(
        package_name='cv2',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred OpenCV version yourself or "
            "re-install the nicr-mt-scene-analysis package with the "
            "additional 'withopencv' target to install a default version, "
            "i.e., `pip install nicr-mt-scene-analysis[withopencv]`."
        )
    )

def is_torch_available(
    raise_error: bool,
    min_version: Optional[str] = None
) -> bool:
    return is_package_available(
        package_name='torch',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred PyTorch version yourself or "
            "re-install the nicr-mt-scene-analysis package with the "
            "additional 'withtorch' target to install a default version, "
            "i.e., `pip install nicr-mt-scene-analysis[withtorch]`."
        )
    )

def is_torchvision_available(
    raise_error: bool = True,
    min_version: Optional[str] = None
) -> bool:
    return is_package_available(
        package_name='torchvision',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred TorchVision version yourself or "
            "re-install the nicr-mt-scene-analysis package with the "
            "additional 'withtorch' target to install a default version, "
            "i.e., `pip install nicr-mt-scene-analysis[withtorch]`."
        )
    )

def is_torchmetrics_available(
    raise_error: bool = True,
    min_version: Optional[str] = None
) -> bool:
    return is_package_available(
        package_name='torchmetrics',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred TorchMetrics version yourself or "
            "re-install the nicr-mt-scene-analysis package with the "
            "additional 'withtorch' target to install a default version, "
            "i.e., `pip install nicr-mt-scene-analysis[withtorch]`."
        )
    )

def install_nicr_multitask_scene_analysis_dependency_import_hooks():
    sys.meta_path.insert(
        0,
        DependencyImportHook(
            module_handlers={
                "opencv": lambda: is_opencv_available(raise_error=True),
                "torch": lambda: is_torch_available(raise_error=True),
                "torchvision": lambda: is_torchvision_available(raise_error=True),
                "torchmetrics": lambda: is_torchmetrics_available(raise_error=True)
            }
        )
    )
