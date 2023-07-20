import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from mmseg.utils import split_images

from ..builder import PIPELINES
from .transforms import PhotoMetricDistortion, Normalize, CLAHE


@PIPELINES.register_module()
class PhotoMetricDistortionMultiImages(PhotoMetricDistortion):
    """ Apply photometric distortion to multiple images sequentially, see class PhotoMetricDistortion for detail
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        img1 = super().__call__(result)['img']

        result = dict(img=img2)
        img2 = super().__call__(result)['img']

        results['img'] = np.concatenate((img1, img2), axis=-1)

        return results


@PIPELINES.register_module()
class NormalizeMultiImages(Normalize):
    """Normalize multiple images, see class Normalize for detail
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        result = super().__call__(result)
        img1 = result['img']
        img_norm_cfg = result['img_norm_cfg']

        result = dict(img=img2)
        img2 = super().__call__(result)['img']

        results['img'] = np.concatenate((img1, img2), axis=-1)
        results['img_norm_cfg'] = img_norm_cfg

        return results


@PIPELINES.register_module()
class CLAHEMultiImages(CLAHE):
    """CLAHE multiple images, see class CLAHE for detail
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        img1 = super().__call__(result)['img']

        result = dict(img=img2)
        img2 = super().__call__(result)['img']

        results['img'] = np.concatenate((img1, img2), axis=-1)

        return results