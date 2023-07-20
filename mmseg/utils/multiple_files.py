import os
import os.path as osp

import numpy as np
from torch import Tensor
import torch
from mmcv.parallel.data_container import DataContainer


def split_images(x):
    if x.ndim == 4:
        channels = x.shape[1]
        channels //= 2
        x1 = x[:, 0:channels, :, :]
        x2 = x[:, channels:, :, :]
    elif x.ndim == 3:
        channels = x.shape[-1]
        channels //= 2
        x1 = x[..., 0:channels]
        x2 = x[..., channels:]
    else:
        raise ValueError(f'dimension of x should be 3 or 4, but got {x.ndim}')

    return x1, x2


def visualize_multiple_images(x, dst_path, channel_per_image=3):
    dtype = 'numpy'
    if isinstance(x, DataContainer):
        x = x.data
        dtype = 'torch'

    # add to 4D tensor, in shape of [batch, height, width, channel]
    if x.ndim < 3:
        x = x[None, ..., None]
    elif x.ndim == 3:
        x = x[None, ...]
    elif x.ndim == 4 and dtype == 'torch':
        x = x.permute((0, 2, 3, 1))

    # if dtype=='torch':
    #     x = x.permute((0, 2, 3, 1))

    # change to numpy
    if isinstance(x, Tensor):
        x = x.numpy()

    c = x.shape[-1]
    try:
        assert c % channel_per_image == 0, f'channels of image ({c}) must be divisible to by {channel_per_image}'
    except:
        pass
    root, ext = osp.splitext(dst_path)
    for kk in range(x.shape[0]):
        for ii in range(c // channel_per_image):
            img = x[kk, :, :, channel_per_image * ii:(ii + 1) * channel_per_image].squeeze()
            img_path = root + f'_{kk}_{ii}' + ext
            save_image_by_cv2(img, dst_path=img_path, if_norm=True, is_bgr=True)
