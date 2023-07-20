import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom_cd import CustomDatasetCD


@DATASETS.register_module()
class LEVIRPlusDataset(CustomDatasetCD):
    """LEVIR+ dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'change')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

@DATASETS.register_module()
class DSIFNDataset(CustomDatasetCD):
    """LEVIR+ dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'change')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
