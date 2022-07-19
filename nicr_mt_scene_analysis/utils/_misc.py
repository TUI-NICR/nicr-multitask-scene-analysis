# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import sys
import os
from functools import partialmethod

from functools import lru_cache


@lru_cache()
def partial_class(cls, *args, **kwargs):
    # modified version of: https://stackoverflow.com/a/38911383
    if args or kwargs:

        class PartialClass(cls):
            __init__ = partialmethod(cls.__init__, *args, **kwargs)

        return PartialClass
    else:
        return cls


class HiddenPrints:
    # Taken from: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def mute():
    # Taken from: https://stackoverflow.com/questions/30829924/suppress-output-in-multiprocessing-process
    sys.stdout = open(os.devnull, 'w')
