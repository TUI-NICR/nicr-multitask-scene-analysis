# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from skimage import data
import torch
import torchvision.transforms as tv_trans

from nicr_mt_scene_analysis.data.preprocessing import AppliedPreprocessingMeta
from nicr_mt_scene_analysis.data.preprocessing import get_applied_preprocessing_meta

from nicr_mt_scene_analysis.data.preprocessing import CloneEntries
from nicr_mt_scene_analysis.data.preprocessing import FlatCloneEntries
from nicr_mt_scene_analysis.data.preprocessing import InstanceClearStuffIDs
from nicr_mt_scene_analysis.data.preprocessing import InstanceTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import MultiscaleSupervisionGenerator
from nicr_mt_scene_analysis.data.preprocessing import NormalizeRGB
from nicr_mt_scene_analysis.data.preprocessing import NormalizeDepth
from nicr_mt_scene_analysis.data.preprocessing import OrientationTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import RandomCrop
from nicr_mt_scene_analysis.data.preprocessing import RandomHorizontalFlip
from nicr_mt_scene_analysis.data.preprocessing import RandomHSVJitter
from nicr_mt_scene_analysis.data.preprocessing import RandomResize
from nicr_mt_scene_analysis.data.preprocessing import Resize
from nicr_mt_scene_analysis.data.preprocessing import ScaleDepth
from nicr_mt_scene_analysis.data.preprocessing import SemanticClassMapper
from nicr_mt_scene_analysis.data.preprocessing import ToTorchTensors
from nicr_mt_scene_analysis.data.preprocessing import TorchTransformWrapper

from nicr_mt_scene_analysis.data.preprocessing.rgb import adjust_hsv
from nicr_mt_scene_analysis.data.preprocessing.utils import _get_relevant_spatial_keys
from nicr_mt_scene_analysis.testing.dataset import get_dataset
from nicr_mt_scene_analysis.testing.preprocessing import get_dummy_sample
from nicr_mt_scene_analysis.testing.preprocessing import show_results
from nicr_mt_scene_analysis.utils import np_biternion2rad


def _check_applied_preprocessing_meta(sample, preprocessor, keys_to_exist=None):
    pre_key = '_applied_preprocessing'
    assert pre_key in sample

    for meta in sample[pre_key]:
        if meta['type'] != preprocessor.__class__.__name__:
            continue

        for key in (keys_to_exist or []):
            assert key in meta
        break


@pytest.mark.parametrize('keys_to_clone', (None, ('rgb', 'depth')))
def test_cloneentries(keys_to_clone):
    """Test CloneEntries"""
    pre = CloneEntries(keys_to_clone=keys_to_clone)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    if keys_to_clone is None:
        keys_to_clone = tuple(sample.keys())

    assert len(sample_pre) == len(sample) + 1    # +1 for clone
    assert len(sample_pre[pre.clone_key]) == len(keys_to_clone)

    clone = sample_pre[pre.clone_key]
    for key in keys_to_clone:
        assert key in clone

        entry = sample[key]
        entry_clone = clone[key]

        if isinstance(entry, np.ndarray):
            assert id(entry_clone.data) != id(entry.data)
            assert np.equal(entry_clone, entry).all()
        elif isinstance(entry, (int, str)):
            # yeah it is python, we are not able to check this
            pass
        else:
            assert id(entry_clone) != id(entry)
            assert entry_clone == entry

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('clone_key', 'ignore_missing_keys', 'cloned_keys')
    )


@pytest.mark.parametrize('keys_to_clone', (None, ('rgb', 'depth')))
@pytest.mark.parametrize('prefix_and_suffix', ((None, '_xyz'), ('abc_', None),
                                               ('abc_', '_xyz')))
def test_flatcloneentries(keys_to_clone, prefix_and_suffix):
    """Test FlatCloneEntries"""
    prefix, suffix = prefix_and_suffix

    pre = FlatCloneEntries(
        keys_to_clone=keys_to_clone,
        key_prefix=prefix,
        key_suffix=suffix
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    if keys_to_clone is None:
        keys_to_clone = tuple(sample.keys())

    assert len(sample_pre) == len(sample) + len(keys_to_clone)

    for key in keys_to_clone:
        key_clone = (prefix or '') + key + (suffix or '')

        assert key_clone in sample_pre

        entry = sample[key]
        entry_clone = sample_pre[key_clone]

        if isinstance(entry, np.ndarray):
            assert id(entry_clone.data) != id(entry.data)
            assert np.equal(entry_clone, entry).all()
        elif isinstance(entry, (int, str)):
            # yeah it is python, we are not able to check this
            pass
        else:
            assert id(entry_clone) != id(entry)
            assert entry_clone == entry

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('key_prefix', 'key_suffix', 'ignore_missing_keys',
                       'added_keys')
    )


def test_instanceclearstuffids():
    """Test InstanceClearStuffIDs"""
    thing_classes = [1, 2, 3]
    classes_is_thing = [True if i in thing_classes else False
                        for i in range(41)]
    pre = InstanceClearStuffIDs(
        semantic_classes_is_thing=classes_is_thing
    )
    # apply preprocessor
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # verify that all stuff pixels have instance id 0
    for c in classes_is_thing[1:]:    # skip void
        if c:
            continue
        # stuff class
        assert (sample_pre['instance'][sample_pre['semantic'] == c] == 0).all()

    # some simple checks
    assert 'instance' in sample_pre

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('stuff_semantic_classes', 'cleared_instance_pixels')
    )

    show_results(sample, sample_pre, 'InstanceClearStuffIDs')


def test_instancetargetgenerator():
    """Test InstanceTargetGenerator"""
    relevant_semantic_classes = [1, 2, 3]    # see get_dummy_sample
    classes = [True if i in relevant_semantic_classes else False
               for i in range(41)]
    pre = InstanceTargetGenerator(
        sigma=8,
        semantic_classes_is_thing=classes
    )

    # get sample (and clear instance ids for stuff classes -> get_dummy_sample)
    sample = get_dummy_sample()
    sample = InstanceClearStuffIDs(semantic_classes_is_thing=classes)(sample)

    # apply preprocessor
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert 'instance_center' in sample_pre
    assert 'instance_offset' in sample_pre
    assert 'instance_foreground' in sample_pre

    # check foreground mask
    assert np.all(
        np.equal(sample_pre['instance_foreground'],
                 np.isin(sample['semantic'], relevant_semantic_classes))
    )
    # more checks are quite complicated, you should consider visual inspection
    # first

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('sigma_for_downscales',
                       'thing_semantic_classes', 'stuff_semantic_classes',
                       'normalized_offset',
                       'encoded_instances', 'skipped_instances_due_to_stuff')
    )

    show_results(sample, sample_pre, 'InstanceTargetGenerator')


@pytest.mark.parametrize('downscales', ((2,), (4, 8), (8, 16, 32)))
def test_multiscale(downscales):
    """Test MultiscaleSupervisionGenerator"""
    pre_1 = MultiscaleSupervisionGenerator(
        downscales=downscales,
        keys=('instance', 'semantic', 'orientations')
    )
    pre_2 = SemanticClassMapper(
        classes_to_map=(1, 10),    # see get_dummy_samples
        new_label=0,
        multiscale_processing=True,
        disable_stats=False
    )
    pre = tv_trans.Compose([pre_1, pre_2])

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # defined at second place
    downscale_fmt = '_down_{}'

    # some simple checks
    for ds in downscales:
        ms_key = downscale_fmt.format(ds)

        assert ms_key in sample_pre

        ms_sample = sample_pre[ms_key]
        for key in ms_sample:
            assert key in sample

        for key in _get_relevant_spatial_keys(ms_sample):
            assert ms_sample[key].shape[0] == int(sample[key].shape[0] / ds)

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre_1,
        keys_to_exist=('downscales', 'keys', 'shapes')
    )
    _check_applied_preprocessing_meta(
        sample_pre, pre_2,
        keys_to_exist=[downscale_fmt.format(ds) for ds in downscales]
    )

    show_results(sample, sample_pre,
                 f'MultiscaleSupervisionGenerator: scales: {downscales}')


def test_orientationtargetgenerator():
    """Test OrientationTargetGenerator"""
    relevant_semantic_classes = [1, 2, 3]
    classes = [True if i in relevant_semantic_classes else False
               for i in range(40)]
    pre = OrientationTargetGenerator(
        semantic_classes_estimate_orientation=classes
    )

    # get dummy sample
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))
    valid_instance_ids = [1, 2]     # see _get_dummy_sample

    # some simple checks
    assert 'orientation' in sample_pre
    assert 'orientation_foreground' in sample_pre
    assert 'orientations_present' in sample_pre
    o = sample_pre['orientation']
    o_f = sample_pre['orientation_foreground']
    i = sample_pre['instance']
    assert np.equal(np.unique(i[o_f]), valid_instance_ids).all()

    for id_ in valid_instance_ids:
        o_id_rad = np_biternion2rad(o[i == id_])
        deg_rad = sample['orientations'][id_]
        assert np.allclose(o_id_rad, deg_rad)

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('semantic_classes',)
    )

    show_results(sample, sample_pre, 'OrientationTargetGenerator')


def test_normalize_rgb():
    """Test NormalizeRGB"""
    dtype = 'float32'
    rgb_mean = np.array((0.485, 0.456, 0.406), dtype=dtype) * 255
    rgb_std = np.array((0.229, 0.224, 0.225), dtype=dtype) * 255

    pre = NormalizeRGB(output_dtype=dtype)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == dtype

    # check rgb values
    for c_idx in range(3):
        c = sample['rgb'][..., c_idx]
        c_pre = sample_pre['rgb'][..., c_idx]
        c_ref = (c.astype(dtype) - rgb_mean[c_idx]) / rgb_std[c_idx]
        assert np.equal(c_pre, c_ref).all()

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('rgb_mean', 'rgb_std', 'output_dtype')
    )

    show_results(sample, sample_pre, 'NormalizeRGB')


@pytest.mark.parametrize('raw_depth', (False, True))
def test_normalize_depth(raw_depth):
    """Test NormalizeDepth"""
    dtype = 'float32'
    depth_mean = np.array(2022.2202, dtype=dtype)
    depth_std = np.array(123.321, dtype=dtype)
    invalid_depth_value = 0.0

    pre = NormalizeDepth(
        depth_mean=depth_mean,
        depth_std=depth_std,
        raw_depth=raw_depth,
        invalid_depth_value=invalid_depth_value,
        output_dtype=dtype
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['depth'].dtype == dtype

    # check depth values
    depth = sample['depth']
    depth_pre = sample_pre['depth']
    mask = depth != invalid_depth_value if raw_depth else slice(None)
    depth_ref = depth.astype(dtype)
    depth_ref[mask] = (depth_ref[mask] - depth_mean) / depth_std
    assert np.equal(depth_pre, depth_ref).all()

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('depth_mean', 'depth_std', 'raw_depth',
                       'invalid_depth_value', 'output_dtype')
    )

    show_results(sample, sample_pre,
                 f'NormalizeDepth: raw_depth: {raw_depth}, {dtype}')


@pytest.mark.parametrize('height', (100, 200, 1000))
@pytest.mark.parametrize('width', (100, 200, 1000))
def test_resize(height, width):
    """Test Resize"""
    pre = Resize(
        height=height,
        width=width,
        keep_aspect_ratio=False
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert sample_pre['some_mask'].dtype == sample['some_mask'].dtype
    assert sample_pre['rgb'].shape == (height, width, 3)
    assert sample_pre['depth'].shape == (height, width)
    assert sample_pre['some_mask'].shape == (height, width)
    # check interpolation
    assert np.equal(sample_pre['rgb'],
                    cv2.resize(sample['rgb'], (width, height),
                               interpolation=cv2.INTER_LINEAR)).all()
    assert np.equal(sample_pre['depth'],
                    cv2.resize(sample['depth'], (width, height),
                               interpolation=cv2.INTER_NEAREST)).all()
    assert np.equal(sample_pre['semantic'],
                    cv2.resize(sample['semantic'], (width, height),
                               interpolation=cv2.INTER_NEAREST)).all()
    assert np.equal(sample_pre['instance'],
                    cv2.resize(sample['instance'], (width, height),
                               interpolation=cv2.INTER_NEAREST)).all()
    assert np.equal(sample_pre['some_mask'],
                    cv2.resize(sample['some_mask'].astype('uint8'),
                               (width, height),
                               interpolation=cv2.INTER_NEAREST) > 0).all()

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('keys_to_ignore',
                       'old_height', 'old_width', 'new_height', 'new_width',
                       'valid_region_slice_y', 'valid_region_slice_x')
    )

    show_results(sample, sample_pre, f'Resize: h: {height}, w: {width}')


@pytest.mark.parametrize('height_width', ((100, 50), (50, 100), (100, 100)))
@pytest.mark.parametrize('padding_mode', ('zero', 'reflect'))
def test_resize_keep_aspect_ratio(height_width, padding_mode):
    """Test Resize with keep_aspect_ratio=True"""
    height, width = height_width
    pre = Resize(
        height=height,
        width=width,
        keep_aspect_ratio=True,
        padding_mode=padding_mode
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert sample_pre['some_mask'].dtype == sample['some_mask'].dtype
    assert sample_pre['rgb'].shape == (height, width, 3)
    assert sample_pre['depth'].shape == (height, width)
    assert sample_pre['some_mask'].shape == (height, width)
    # check interpolation (done in test_resize)

    # TODO: add some meaningful checks

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('keys_to_ignore',
                       'old_height', 'old_width', 'new_height', 'new_width',
                       'valid_region_slice_y', 'valid_region_slice_x')
    )

    show_results(sample, sample_pre,
                 f'Resize while keeping aspect ratio: h: {height}, w: {width}', True)


@pytest.mark.parametrize('random_input', (False, True))
def test_resize_uint32(random_input):
    """Test Resize with uint32 input, i.e., panoptic segmentation"""

    # we use OpenCV for resizing, however, OpenCV does not support uint32
    # inputs, which is a problem for the panoptic segmentation ('panoptic' key),
    # we circumvent this limitation by viewing the uint32 input as uint8 input,
    # i.e, by converting from single-channel uint32 (grayscale uint32) to
    # 4-channel uint8 (rgba uint8)
    # note: this workaround is only possible for nearest interpolation!

    # create preprocessor
    height, width = 101, 150
    pre = Resize(height=height, width=width)

    # prepare sample
    if not random_input:
        sample = get_dummy_sample()
    else:
        h_w = (12, 13)
        semantic = np.random.randint(
            low=0, high=np.iinfo('uint16').max + 1,
            size=h_w,
            dtype='uint16'
        )
        instance = np.random.randint(
            low=0, high=np.iinfo('uint16').max + 1,
            size=h_w,
            dtype='uint16'
        )
        rgb = np.random.randint(0, 255 + 1, (*h_w, 3), dtype='uint8')
        sample = {
            'rgb': rgb,   # required for shape stuff
            'semantic': semantic,
            'instance': instance
        }

    # shorter names
    semantic = sample['semantic']
    instance = sample['instance']

    # prepare panoptic key
    shift = 2**16
    panoptic = semantic.astype('uint32') * shift + instance
    sample['panoptic'] = panoptic

    # apply preprocessor
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['panoptic'].shape == (height, width)
    assert sample_pre['panoptic'].dtype == 'uint32'

    # check values
    np.testing.assert_array_equal(
        np.unique(sample_pre['panoptic']),
        np.unique(sample['panoptic'])
    )
    np.testing.assert_array_equal(
        np.unique(sample_pre['panoptic'] // shift),
        np.unique(semantic)
    )
    np.testing.assert_array_equal(
        np.unique(sample_pre['panoptic'] % shift),
        np.unique(instance)
    )

    # check applied preprocessing meta (done in test_resize)

    show_results(sample, sample_pre, 'Resize with uint32 input')


@pytest.mark.parametrize('min_scale', (0.8, 1.0))
@pytest.mark.parametrize('max_scale', (1.0, 1.4))
def test_randomresize(min_scale, max_scale):
    """Test RandomResize"""
    pre = RandomResize(min_scale=min_scale,
                       max_scale=max_scale)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    # note that results are checked in test_resize
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert sample_pre['some_mask'].dtype == sample['some_mask'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('min_scale', 'max_scale', 'keys_to_ignore',
                       'old_height', 'old_width', 'new_height', 'new_width',
                       'valid_region_slice_y', 'valid_region_slice_x')
    )

    show_results(sample, sample_pre,
                 f'RandomResize: min: {min_scale}, max: {max_scale}')


@pytest.mark.parametrize('crop_height', (100, 200, 1000))
@pytest.mark.parametrize('crop_width', (100, 200, 1200))
def test_randomcrop(crop_height, crop_width):
    """Test RandomCrop"""
    pre = RandomCrop(crop_height=crop_height,
                     crop_width=crop_width)

    # apply augmentation
    sample = get_dummy_sample()  # shape: 256x256x3
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    #
    if 1000 == crop_height or 1200 == crop_width:
        # crop is larger than image, should be resized before
        meta = get_applied_preprocessing_meta(sample_pre)
        assert meta[0]['was_resized']  # 0 -> first and only preprocessor

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('crop_height', 'crop_width',
                       'was_resized', 'resize_height', 'resize_width',
                       'crop_slice_y', 'crop_slice_x')
    )

    show_results(sample, sample_pre,
                 f'RandomCrop: h: {crop_height}, w: {crop_width}')


@pytest.mark.parametrize('p', (0.1, 0.5, 0.8, 1.0))
def test_randomhorizontalflip(p):
    """Test RandomHorizontalFlip"""
    pre = RandomHorizontalFlip(p=p)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    if 1.0 == p:
        for k in _get_relevant_spatial_keys(sample):
            assert np.equal(sample_pre[k], sample[k][:, ::-1]).all()
        for id_ in sample['orientations']:
            angle = np.rad2deg(sample['orientations'][id_])
            angle_ref = (360 - angle) % 360

            assert np.deg2rad(angle_ref) == sample_pre['orientations'][id_]

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('p', 'was_flipped')
    )

    show_results(sample, sample_pre, f'RandomFlip: p: {p}')


@pytest.mark.parametrize('hue_jitter', (10/(360./2),))    # +-10 degrees
@pytest.mark.parametrize('saturation_jitter', (20/(255.),))    # +- ~8%
@pytest.mark.parametrize('value_jitter', (50/(255.),))    # +- ~16%
def test_hsvjitter(hue_jitter, saturation_jitter, value_jitter):
    """Test RandomHSVJitter"""
    pre = RandomHSVJitter(hue_jitter=hue_jitter,
                          saturation_jitter=saturation_jitter,
                          value_jitter=value_jitter)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('hue_limits', 'saturation_limits', 'value_limits',
                       'applied_hue_offset', 'applied_saturation_offset',
                       'applied_value_offset')
    )

    show_results(sample, sample_pre,
                 f'RandomHSVJitter: h: {hue_jitter*180}, '
                 f's: {saturation_jitter*255}, v: {value_jitter*255}')


@pytest.mark.parametrize('img', (data.astronaut(), data.chelsea(), data.coffee()))
@pytest.mark.parametrize('hue_jitter_offset', (10//2, -10//2))
@pytest.mark.parametrize('saturation_jitter_offset', (20, -20))
@pytest.mark.parametrize('value_jitter_offset', (50, -50))
def test_hsvjitter_limits(img,
                          hue_jitter_offset,
                          saturation_jitter_offset,
                          value_jitter_offset):
    """Test to visualize limits of HSV color jittering"""
    # adjust color in hsv space
    img_pre = adjust_hsv(img,
                         hue_jitter_offset,
                         saturation_jitter_offset,
                         value_jitter_offset)

    show_results({'rgb': img}, {'rgb': img_pre},
                 f'HSVJitterLimits: h: {hue_jitter_offset*2}, '
                 f's: {saturation_jitter_offset}, v: {value_jitter_offset}')


def test_semanticclassmapper():
    """Test SemanticClassMapper"""
    classes_to_map = (1, 10)    # see get_dummy_samples
    new_label = 0
    pre = SemanticClassMapper(
        classes_to_map=classes_to_map,
        new_label=new_label,
        disable_stats=False
    )

    # get sample
    sample = get_dummy_sample()

    # apply preprocessor
    sample_pre = pre(deepcopy(sample))

    # check that mapping was applied
    mask_changed = np.zeros_like(sample['semantic'], dtype='bool')
    for c in classes_to_map:
        mask = sample['semantic'] == c
        assert (sample_pre['semantic'][mask] == new_label).all()

        mask_changed = np.logical_or(mask_changed, mask)

    # check that remaining pixels are unchanged
    mask = np.logical_not(mask_changed)
    assert (sample_pre['semantic'][mask] == sample['semantic'][mask]).all()

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('semantic_classes_to_map', 'new_label', 'disable_stats',
                       'mapped_pixels')
    )

    show_results(sample, sample_pre, 'SemanticClassMapper')


def test_totorchtensors():
    """Test ToTorchTensors"""
    pre = ToTorchTensors()

    # apply augmentation
    sample = get_dummy_sample()

    # pytorch does not support uint16 as used for depth, we circumvent this by
    # manually casting the depth image to float, in chained processing, this
    # is usually done by NormalizeDepth
    sample['depth'] = sample['depth'].astype('float32')

    sample_pre = pre(deepcopy(sample))

    # some simple checks
    for k, v in sample_pre.items():
        if isinstance(v, (dict, str, int)):
            # primitive types should be unchanged
            assert sample[k] == v
            continue

        if isinstance(v, AppliedPreprocessingMeta):
            # skip meta
            continue

        assert isinstance(v, torch.Tensor)

    # check applied preprocessing meta (there are no keys to check)
    _check_applied_preprocessing_meta(sample_pre, pre)


@pytest.mark.parametrize('interpolation', (tv_trans.InterpolationMode.NEAREST,
                                           tv_trans.InterpolationMode.BILINEAR))
def test_torchtransformwrapper(interpolation):
    """Test TorchTransformWrapper"""

    # only works for nearest interpolation
    if interpolation == tv_trans.InterpolationMode.BILINEAR:
        pytest.xfail('should not work for bilinear interpolation')

    # preprocessor that wraps multiple transforms
    pre = TorchTransformWrapper(
        tv_trans.Compose([
            tv_trans.RandomResizedCrop(10, interpolation=interpolation),
            tv_trans.TenCrop(5)
        ])
    )

    # prepare sample
    h_w = (16, 16)
    dtype = torch.float32
    rgb = torch.arange(0, 256, dtype=dtype).reshape(1, *h_w).repeat(3, 1, 1)
    depth = rgb[0, ...].unsqueeze(0) * 100
    sample = {'rgb': rgb, 'depth': depth}

    # apply preprocessing
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].shape == (10, 3, 5, 5)
    assert sample_pre['depth'].shape == (10, 1, 5, 5)
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert torch.allclose(
        sample_pre['rgb'][:, 0, :, :].unsqueeze(1),
        sample_pre['depth'].type(sample_pre['rgb'].dtype) / 100
    )

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('transform_obj', 'keys')
    )


def test_torchtransformwrapper_qualitative():
    # second just qualitative test with real images

    class EnsureFloat:
        def __call__(self, sample):
            for key in _get_relevant_spatial_keys(sample):
                sample[key] = sample[key].astype(np.float32)
            return sample

    pre = tv_trans.Compose([
        EnsureFloat(),
        ToTorchTensors(),
        # wrap two single transforms
        TorchTransformWrapper(
            tv_trans.RandomResizedCrop(
                300, interpolation=tv_trans.InterpolationMode.NEAREST
            )
        ),
        TorchTransformWrapper(tv_trans.FiveCrop(150))
    ])

    dataset = get_dataset(
        name='nyuv2',
        split='train',
        sample_keys=('rgb', 'depth')
    )

    crop_order = ('top_left', 'top_right', 'bottom_left', 'bottom_right',
                  'center')
    for i in range(3):  # first 3 samples
        sample = dataset[i]
        sample_pre = pre(deepcopy(sample))

        for crop_rgb, crop_depth, crop_pos in zip(sample_pre['rgb'],
                                                  sample_pre['depth'],
                                                  crop_order):
            crop_rgb = crop_rgb.numpy().transpose(1, 2, 0).astype(np.uint8)
            crop_depth = crop_depth.numpy().transpose(1, 2, 0).astype(np.uint16)
            show_results(
                sample, {'rgb': crop_rgb, 'depth': crop_depth},
                f'TorchTransformWrapper sample {i} (rgb & depth) - {crop_pos}')


@pytest.mark.parametrize('min_max', ((0.0, 1.0), (-1.0, 1.0)))
@pytest.mark.parametrize('raw_depth', (False, True))
def test_scale_depth(min_max, raw_depth):
    """Test ScaleDepth"""
    dtype = 'float32'
    new_min, new_max = min_max
    invalid_depth_value = 0.0

    pre = ScaleDepth(
        new_min=new_min,
        new_max=new_max,
        raw_depth=raw_depth,
        invalid_depth_value=invalid_depth_value,
        output_dtype=dtype
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['depth'].dtype == dtype

    # check depth values
    depth = sample['depth']
    depth_pre = sample_pre['depth']
    mask_valid = depth != invalid_depth_value if raw_depth else slice(None)
    depth_pre[mask_valid].min() == new_min
    depth_pre[mask_valid].max() == new_max
    if raw_depth:
        raw_depth_mask = depth == invalid_depth_value
        assert (depth_pre[raw_depth_mask] == invalid_depth_value).all()

    # check applied preprocessing meta
    _check_applied_preprocessing_meta(
        sample_pre, pre,
        keys_to_exist=('new_min', 'new_max', 'raw_depth',
                       'invalid_depth_value', 'output_dtype')
    )

    show_results(sample, sample_pre,
                 f'ScaleDepth: min_max: {min_max}, raw_depth: {raw_depth}, '
                 f'{dtype}')
