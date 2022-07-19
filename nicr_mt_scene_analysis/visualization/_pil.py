# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from PIL import Image


def to_pil_img(img: np.array, palette=None) -> Image.Image:
    # note that OpenCV is not able to handle indexed pngs correctly.
    if img.max() > 255:
        dtype = 'uint16'
    else:
        dtype = 'uint8'
        if palette is not None:
            palette = palette[:256]

    img = Image.fromarray(img.astype(dtype))
    if palette is not None:
        img.putpalette(list(np.asarray(palette, dtype=dtype).flatten()))

    return img
