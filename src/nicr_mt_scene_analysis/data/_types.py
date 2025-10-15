# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""


class CollateIgnoredDict(dict):
    """Enables a custom pytorch collate function ignore this dict."""
    pass


class PreprocessingParameterDict(CollateIgnoredDict):
    """Enables a custom PyTorch collate function ignore this dict."""
    pass


class AppliedPreprocessingMeta(list):
    """Enables a custom PyTorch collate function ignore this type."""
    pass
