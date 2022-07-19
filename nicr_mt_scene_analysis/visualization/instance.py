# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Optional, Tuple, Union

import os

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from ..utils import np_rad2deg
from ..utils import np_biternion2deg
from ._pil import to_pil_img


class ColorMapper:
    def __init__(self):
        # TODO: nicer colormap that really has more than 256 colors
        # hacky way to have more than 256 colors
        default_cmap = self._get_simple_colormap(256) * 4
        self.default_cmap = np.array(default_cmap, dtype='uint8')
        self.id_to_color = {0: (0, 0, 0)}
        # index 0 defaults to black color, as it is typically void
        self.colormap_np = np.array([[0, 0, 0]], dtype='uint8')

    def __call__(self, id_):
        # build colormap_np
        if id_ >= self.colormap_np.shape[0]:
            # add some rows with black color as default
            n_additional_rows = id_ - self.colormap_np.shape[0] + 1
            additional_rows = np.array([(0, 0, 0)] * n_additional_rows, dtype='uint8')
            self.colormap_np = np.vstack((self.colormap_np, additional_rows))

        # build dictionary
        if id_ not in self.id_to_color:
            # take the next color of the default_cmap
            color = self.default_cmap[len(self.id_to_color)]
            # set current color
            self.colormap_np[id_] = color
            self.id_to_color[id_] = tuple(int(c) for c in color)

        return self.id_to_color[id_]

    def add_colors_for_all_ids_in_image(self, img):
        for id_ in np.unique(img):
            self(id_)

    def get_colored_image(self, img):
        self.add_colors_for_all_ids_in_image(img)
        return self.colormap_np[img]

    @staticmethod
    def _get_simple_colormap(n: int) -> Tuple[Tuple[int, int, int]]:
        # n <= 256
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = []
        for i in range(n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap.append((r, g, b))

        return tuple(cmap)


_INSTANCE_COLOR_MAPPER = ColorMapper()


def visualize_instance_center(
    center_img: np.array,
    min_: Optional[int] = None,
    max_: Optional[int] = None,
) -> np.array:
    assert center_img.ndim == 2

    if np.issubdtype(center_img.dtype, np.floating):
        # instance center given as heatmap
        # default (gt) range for instance centers is [0, 1]
        min_ = min_ or 0
        max_ = max_ or 1

        center_img_01 = (center_img - min_) / (max_ - min_)

        return np.clip(center_img_01*255, 0, 255).astype('uint8')

    # instance center given as mask
    img = np.zeros_like(center_img, dtype='uint8')
    for y, x in zip(*center_img.nonzero()):
        img = cv2.drawMarker(img, position=(x, y), color=255,
                             markerType=cv2.MARKER_TILTED_CROSS,
                             thickness=2, markerSize=15)
    return img


def visualize_instance_center_pil(
    center_img: np.array,
    min_: Optional[int] = None,
    max_: Optional[int] = None,
) -> Image.Image:
    return to_pil_img(
        visualize_instance_center(center_img, min_, max_),
        palette=None
    )


def visualize_instance_offset(offset_img: np.array,
                              foreground_mask: Union[None, np.array] = None) -> np.array:
    assert offset_img.ndim == 3
    assert offset_img.shape[-1] == 2

    offset_vec_img = np.zeros((offset_img.shape[0], offset_img.shape[1], 3),
                              dtype=np.uint8)
    # set V of HSV image
    offset_vec_img[:, :, 2] = 255

    dy_img = np.array(offset_img[..., 0], dtype='float32')
    dx_img = np.array(offset_img[..., 1], dtype='float32')
    norm = np.sqrt(dx_img**2 + dy_img**2)
    dy_img = np.divide(dy_img, norm, where=(norm != 0))
    dx_img = np.divide(dx_img, norm, where=(norm != 0))
    max_abs = norm.max()
    vec_ang = (np.rad2deg(np.arctan2(-dy_img, dx_img)) + 180)/2
    vec_ang = (vec_ang - 90/2) % (360/2)
    vec_ang = vec_ang.round().astype(np.uint8)
    vec_len = (norm/max_abs*255).round().astype(np.uint8)
    offset_vec_img[:, :, 0] = vec_ang
    offset_vec_img[:, :, 1] = vec_len

    rgb_img = cv2.cvtColor(offset_vec_img, cv2.COLOR_HSV2RGB)
    if foreground_mask is not None:
        rgb_img[foreground_mask == 0] = [255, 255, 255]
    return rgb_img


def visualize_instance_offset_pil(offset_img: np.array,
                                  foreground_mask: Union[None, np.array] = None) -> Image.Image:
    return to_pil_img(visualize_instance_offset(offset_img, foreground_mask),
                      palette=None)


def visualize_instance(instance_img: np.array) -> np.array:
    assert instance_img.ndim == 2

    return _INSTANCE_COLOR_MAPPER.get_colored_image(instance_img)


def visualize_instance_pil(instance_img: np.array) -> Image.Image:
    assert instance_img.ndim == 2

    _INSTANCE_COLOR_MAPPER.add_colors_for_all_ids_in_image(instance_img)

    return to_pil_img(instance_img, palette=_INSTANCE_COLOR_MAPPER.colormap_np)


def visualize_instance_orientations(
    instance_img: np.array,
    orientations: Dict[int, float],
    thickness: int = 2,
    font_size: int = 30,
    bg_color: int = 0,
    bg_color_font: str = 'black',
    draw_outline: bool = True
) -> np.array:
    assert instance_img.ndim == 2

    orientation_img = np.zeros(instance_img.shape+(3,), dtype='uint8') + bg_color

    font_fp = os.path.join(os.path.dirname(__file__), 'FreeMonoBold.ttf')
    font = ImageFont.truetype(font_fp, font_size)

    for instance_id in np.unique(instance_img)[1:]:    # skip void
        if instance_id not in orientations:
            # we do not have an orientation for this instance
            continue

        # get instance mask and center point
        instance_mask = instance_img == instance_id
        ys, xs = np.where(instance_mask)
        c_y = int(np.mean(ys))
        c_x = int(np.mean(xs))

        if draw_outline:
            # get instance contour (uint8 is not supported using floodfill)
            contours, _ = cv2.findContours(instance_mask.astype('int'),
                                           mode=cv2.RETR_FLOODFILL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # draw instance contour
            color = _INSTANCE_COLOR_MAPPER(instance_id)
            orientation_img = cv2.drawContours(orientation_img,
                                               contours=contours, contourIdx=-1,
                                               color=color, thickness=thickness)
        else:
            color = (255, 255, 255)

        # put orientation text (we use pil as OpenCV's puttext is just bad)
        pil_img = Image.fromarray(orientation_img)
        if orientations[instance_id] is not None:
            text = f'{np_rad2deg(orientations[instance_id]):.0f}°'
            w, h = font.getsize(text)
            pil_draw = ImageDraw.Draw(pil_img)
            pil_draw.rectangle((c_x-w//2, c_y-h//2, c_x+w//2, c_y+h//2),
                               fill=bg_color_font)
            pil_draw.text((c_x, c_y),
                          text=text,
                          font=font, fill=color, anchor="mm")
        orientation_img = np.asanyarray(pil_img)

    return orientation_img


def visualize_instance_orientations_pil(
    instance_img: np.array,
    orientations: Dict[int, float],
    thickness: int = 2,
    font_size: int = 30,
    bg_color: int = 0,
    bg_color_font: str = 'black',
    draw_outline: bool = True
) -> Image.Image:
    return to_pil_img(visualize_instance_orientations(instance_img,
                                                      orientations,
                                                      thickness,
                                                      font_size,
                                                      bg_color,
                                                      bg_color_font,
                                                      draw_outline))


def visualize_orientation(orientation_img: np.array) -> np.array:
    assert orientation_img.ndim == 3
    assert orientation_img.shape[-1] == 2

    h, w, _ = orientation_img.shape
    orientation_img_hsv = np.zeros((h, w, 3), dtype='uint8')
    # convert biternion output to deg
    # to use 'np_biternion2deg' we have to transpose (back) to channels first
    # and to a batch axis (channel axis hast to be #2)
    biternion = orientation_img.transpose(2, 0, 1)[None]
    orientation_img_deg = np_biternion2deg(biternion)[0]

    # we use OpenCV for HSV space, thus, hue is in [0, 179] (step size is 2
    # degrees), saturation is in [0, 255], and value is in [0, 255]
    orientation_img_hsv[..., 0] = (orientation_img_deg // 2).astype('uint8')
    orientation_img_hsv[..., 1] = 255     # default saturation
    orientation_img_hsv[..., 2] = 128     # default value

    return cv2.cvtColor(orientation_img_hsv, cv2.COLOR_HSV2RGB)


def visualize_orientation_pil(orientation_img: np.array) -> Image.Image:
    return to_pil_img(visualize_orientation(orientation_img))
