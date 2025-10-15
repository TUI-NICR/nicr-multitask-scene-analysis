# -*- coding: utf-8 -*-
"""
Generische Hilfsfunktion für Dense Decoder (semantic, normal, embedding).
"""
from collections import OrderedDict
from typing import Optional, Type, List
from torch import nn


def create_task_head(
    n_channels_in: int,
    n_channels_out: int,
    upsampling: Optional[Type] = None,
    n_upsamplings: int = 0,
    post_modules: Optional[List[nn.Module]] = None
) -> nn.Module:
    modules = []
    is_main_output = n_upsamplings != 0
    modules.append((
        'conv',
        nn.Conv2d(n_channels_in, n_channels_out,
                  kernel_size=3 if is_main_output else 1,
                  padding=1 if is_main_output else 0)
    ))

    for i in range(n_upsamplings):
        modules.append((
            f'upsample_{i}',
            upsampling(n_channels=n_channels_out)
        ))

    if post_modules is not None:
        for idx, mod in enumerate(post_modules):
            modules.append((f'post_{idx}', mod))

    return nn.Sequential(OrderedDict(modules))
