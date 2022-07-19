# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>

Some parts of this code are based on:
    https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
"""
from typing import Dict, Any

from collections import Counter

import cv2
import numpy as np
from PIL import Image


class IdGenerator():
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.color_counter = Counter()
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_next_hsv_color(self, base, s_step=round(255*0.05), v_step=round(255*0.01)):
        base = np.array(base, dtype=np.uint8)
        base_hsv = cv2.cvtColor(base.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0]

        current_ctr = self.color_counter[str(base_hsv)]
        self.color_counter[str(base_hsv)] += 1
        base_hsv[1] = (base_hsv[1]-s_step*current_ctr) % 255
        base_hsv[2] = (base_hsv[2]-v_step*current_ctr) % 255
        base_rgb = cv2.cvtColor(base_hsv.reshape(1, 1, 3),
                                cv2.COLOR_HSV2RGB)[0][0]
        # base[0] = (base[0] + 10) % 360
        return tuple(base_rgb)

    def get_color(self, cat_id):
        def random_color(base, max_dist=45):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                # color = self.get_next_hsv_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def visualize_panoptic(
    panoptic_img: np.array,
    num_cats: int,
    categorys: Dict[int, Any],
    max_instances: int = 100,
) -> np.array:
    id_generator = IdGenerator(categorys)
    h, w = panoptic_img.shape
    panoptic_result = np.zeros((h, w, 3), dtype=np.uint8)

    uids = np.unique(panoptic_img)
    VOID = 0

    for el in uids:
        sem = el // max_instances
        if sem == VOID:
            continue
        mask = panoptic_img == el
        segment_id, color = id_generator.get_id_and_color(sem)
        panoptic_result[mask] = color

    return panoptic_result


def visualize_panoptic_pil(
    panoptic_img: np.array,
    num_cats: int,
    categorys: Dict[int, Any],
    max_instances: int = 100,
) -> Image.Image:
    panoptic_img = visualize_panoptic(
        panoptic_img,
        num_cats,
        categorys,
        max_instances,
    )
    return Image.fromarray(panoptic_img)
