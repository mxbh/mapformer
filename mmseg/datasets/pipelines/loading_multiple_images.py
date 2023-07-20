
import os.path as osp

import mmcv
import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImagesFromFile(object):
    """Load multiple images from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # image 1
        filename1 = osp.join(results['img1_prefix'],
                                results['img_info']['filename'])
        img1_bytes = self.file_client.get(filename1)
        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)

        # image 2
        filename2 = osp.join(results['img2_prefix'],
                                results['img_info']['filename'])
        img2_bytes = self.file_client.get(filename2)
        img2 = mmcv.imfrombytes(
            img2_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['filename2'] = filename2
        results['ori_filename'] = results['img_info']['filename']
        # results['img'] = torch.cat((img1, img2), dim=0)
        results['img'] = np.concatenate((img1, img2), axis=-1)
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img1.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
