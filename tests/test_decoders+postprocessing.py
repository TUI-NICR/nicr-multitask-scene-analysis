# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmeanu.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.activation import get_activation_class
from nicr_mt_scene_analysis.model.block import get_block_class
from nicr_mt_scene_analysis.model.decoder import SemanticDecoder
from nicr_mt_scene_analysis.model.decoder import InstanceDecoder
from nicr_mt_scene_analysis.model.decoder import NormalDecoder
from nicr_mt_scene_analysis.model.decoder import PanopticHelper
from nicr_mt_scene_analysis.model.decoder import SceneClassificationDecoder
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import get_encoder_decoder_fusion_class
from nicr_mt_scene_analysis.model.normalization import get_normalization_class
from nicr_mt_scene_analysis.model.postprocessing import get_postprocessing_class
from nicr_mt_scene_analysis.model.upsampling import get_upsampling_class
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


def decoders_test(tasks, panoptic_enabled, do_postprocessing, training, debug,
                  tmp_path):
    # set fixed seed (cherry-picked seed)
    # necessary to ensure passing instance tests with random input and weights
    torch.manual_seed(0)

    # common parameters used in almost all decoders
    common_kwargs = {
        'n_channels_in': 512,
        'n_channels': (512, 256, 128),
        'downsampling_in': 32,
        'fusion': get_encoder_decoder_fusion_class('add-rgb'),
        'fusion_n_channels': (256, 128, 64),
        'normalization': get_normalization_class('batchnorm'),
        'activation': get_activation_class('relu'),
        'upsampling': get_upsampling_class('learned-3x3-zeropad'),
        'prediction_upsampling': get_upsampling_class('learned-3x3-zeropad'),
        'block': get_block_class('nonbottleneck1d'),
        'n_blocks': 1,
    }

    if 'semantic' in tasks:
        semantic_decoder = SemanticDecoder(
            n_classes=40,
            postprocessing=get_postprocessing_class('semantic', debug=debug),
            **common_kwargs
        )
        decoder = semantic_decoder
    if 'instance' in tasks:
        instance_decoder = InstanceDecoder(
            n_channels_per_task=32,
            with_orientation=('orientation' in tasks),
            postprocessing=get_postprocessing_class(
                'instance',
                heatmap_threshold=0.1,
                heatmap_nms_kernel_size=3,
                top_k_instances=64,
                debug=debug
            ),
            **common_kwargs
        )
        decoder = instance_decoder

    if panoptic_enabled:
        decoder = PanopticHelper(
            semantic_decoder=semantic_decoder,
            instance_decoder=instance_decoder,
            postprocessing=get_postprocessing_class(
                'panoptic',
                semantic_postprocessing=semantic_decoder.postprocessing,
                instance_postprocessing=instance_decoder.postprocessing,
                semantic_classes_is_thing=(True, )*40,
                semantic_class_has_orientation=(True, )*40,
                debug=debug
            )
        )

    if 'normal' in tasks:
        decoder = NormalDecoder(
            n_channels_out=3,
            postprocessing=get_postprocessing_class('normal', debug=debug),
            **common_kwargs
        )

    if 'scene' in tasks:
        common_kwargs['n_channels_in'] = 512//2
        decoder = SceneClassificationDecoder(
            n_classes=10,
            postprocessing=get_postprocessing_class('scene', debug=debug),
            **common_kwargs
        )

    # set up inputs for decoders
    x = (
        torch.rand(3, 512, 20, 15),    # output of context module
        (torch.rand(3, 512//2, 1, 1),)     # at least one context branch (from GAP)
    )
    skips_reversed = (
        (torch.rand(3, 256, 40, 30), None),     # downsample: 16
        (torch.rand(3, 128, 80, 60), None),    # downsample: 8
        (torch.rand(3, 64, 160, 120), None),    # downsample: 4
    )
    batch = {}
    if 'instance' in tasks:
        # pure instance segmentation task requires gt foreground mask
        batch['instance_foreground'] = torch.ones((3, 480, 640),
                                                  dtype=torch.bool)
    if 'orientation' in tasks:
        # orientation estimation requires a gt segmentation and foreground mask
        batch['instance'] = torch.ones((3, 480, 640), dtype=torch.bool)
        batch['orientation_foreground'] = torch.ones((3, 480, 640),
                                                     dtype=torch.bool)

    # we need at least one fullres key in the batch for shape deriving in
    # inference (validation) postprocessing
    batch['rgb_fullres'] = torch.ones((3, 3, 480, 640), dtype=torch.uint8)

    if not training:
        decoder.eval()

    output = decoder(x, skips_reversed, batch,
                     do_postprocessing=do_postprocessing)

    if not do_postprocessing:
        assert isinstance(output, tuple)
        assert len(output) == 2
    else:
        # postprocessed output of decoder is returned: dict
        assert isinstance(output, dict)
        assert len(output)

        keys = []
        if 'semantic' in tasks or panoptic_enabled:
            keys += ['semantic_output', 'semantic_side_outputs']
            if not training:
                keys += ['semantic_segmentation_score',
                         'semantic_segmentation_idx',
                         'semantic_segmentation_score_fullres',
                         'semantic_segmentation_idx_fullres']

        if 'instance' in tasks or panoptic_enabled:
            keys += ['instance_output', 'instance_side_outputs']
            if not training:
                keys += ['instance_centers',
                         'instance_offsets',
                         'instance_predicted_centers',
                         'instance_segmentation_gt_foreground',
                         'instance_segmentation_gt_foreground_fullres']
                if debug:
                    keys += ['instance_segmentation_all_foreground',
                             'instance_segmentation_all_foreground_fullres']

        if panoptic_enabled:
            if not training:
                keys += ['panoptic_foreground_mask',
                         'panoptic_foreground_mask_fullres',
                         'panoptic_instance_predicted_centers',
                         'panoptic_instance_segmentation',
                         'panoptic_instance_segmentation_fullres',
                         'panoptic_segmentation_deeplab',
                         'panoptic_segmentation_deeplab_fullres',
                         'panoptic_segmentation_deeplab_ids',
                         'panoptic_segmentation_deeplab_semantic_segmentation',
                         'panoptic_segmentation_deeplab_instance_segmentation']
                if 'orientation' in tasks:
                    keys += ['orientations_panoptic_segmentation_deeplab_instance_segmentation']

        if 'orientation' in tasks:
            if not training:
                keys += ['orientations_gt_instance_gt_orientation_foreground',
                         'orientations_instance_segmentation_gt_orientation_foreground']
                if debug:
                    keys += ['orientations_gt_instance',
                             'orientations_instance_segmentation']

        if 'normal' in tasks:
            keys += ['normal_output', 'normal_side_outputs']

        if 'scene' in tasks:
            keys += ['scene_output']
            if not training:
                keys += ['scene_class_score', 'scene_class_idx']

        # check that all keys are present
        for k in keys:
            assert k in list(output.keys())

    # export decoders to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging
        return
    # determine filename and filepath
    tasks_str = '+'.join(tasks)
    if panoptic_enabled:
        tasks_str += '+panoptic'
    filename = f'decoders_{tasks_str}'
    filename += f'__train_{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    x = (x, skips_reversed, batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, decoder, x)


@pytest.mark.parametrize('task', ('semantic',
                                  'instance',
                                  ('instance', 'orientation'),
                                  'normal',
                                  'scene'))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('debug', (False, True))
def test_decoders_single_decoder(task, do_postprocessing, training, debug,
                                 tmp_path):
    """Test in single decoder setup"""
    tasks = task if isinstance(task, tuple) else (task,)
    decoders_test(tasks=tasks,
                  panoptic_enabled=False,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  debug=debug,
                  tmp_path=tmp_path)


@pytest.mark.parametrize('tasks', (('semantic', 'instance'),
                                   ('semantic', 'instance', 'orientation')))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('debug', (False, True))
def test_decoders_panoptic(tasks, do_postprocessing, training, debug,
                           tmp_path):
    """Test in panoptic segmentation setting"""
    decoders_test(tasks=tasks,
                  panoptic_enabled=True,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  debug=debug,
                  tmp_path=tmp_path)
