import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from .. import builder
from ..builder import SEGMENTORS
from ..segmentors.encoder_decoder import EncoderDecoder
from ...core import add_prefix
from ...ops import resize


@SEGMENTORS.register_module()
class EncoderDecoderCCD(EncoderDecoder):
    '''
    Overall model class for Conditional CD.
    '''
    def __init__(self, inference_tile_size=None, *args, **kwargs):
        super(EncoderDecoderCCD, self).__init__(*args, **kwargs)
        self.inference_tile_size = inference_tile_size
        self.tile_inference = self.inference_tile_size is not None

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, gt_semantic_seg_pre=None, gt_semantic_seg_post=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(
                img=img,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg_post=gt_semantic_seg_post, 
                **kwargs
                )
        else:
            return self.forward_test(
                imgs=img,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                **kwargs
            )

    def forward_train(self, img, img_metas, gt_semantic_seg, gt_semantic_seg_pre=None, gt_semantic_seg_post=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            gt_semantic_seg_pre (Tensor): Segmentation mask for t1.
            gt_semantic_seg_post (Tensor): Segmentation mask for t2.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x=x,
            img_metas=img_metas,
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post,
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            #raise RuntimeWarning('This has not been implemented properly')
            loss_aux = self._auxiliary_head_forward_train(
                x=x,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg=gt_semantic_seg,
                gt_semantic_seg_post=gt_semantic_seg_post,
            )
            losses.update(loss_aux)

        return losses

    def forward_test(self, imgs, img_metas, gt_semantic_seg_pre=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            gt_semantic_seg_pre (List[Tensor]): Segmentations for t1.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')
        if gt_semantic_seg_pre is not None and not isinstance(gt_semantic_seg_pre, list):
            raise TypeError(f'gt_semantic_seg_pre must be None or list, got {type(gt_semantic_seg_pre)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas) or num_augs != len(gt_semantic_seg_pre):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(
                img=imgs[0], 
                img_metas=img_metas[0], 
                gt_semantic_seg_pre=gt_semantic_seg_pre[0], 
                **kwargs
            )
        else:
            return self.aug_test(
                imgs=imgs, 
                img_metas=img_metas, 
                **kwargs
            )

    def simple_test(self, img, img_metas, gt_semantic_seg_pre=None, rescale=True):
        """Simple test with single image."""
        output = self.inference(
            img=img, 
            gt_semantic_seg_pre=gt_semantic_seg_pre, 
            img_meta=img_metas, 
            rescale=rescale
        )
        bc = output['bc']
        sem = output['sem']

        bc = bc.argmax(dim=1)
        sem = sem.argmax(dim=1)

        bc = bc.cpu().numpy().astype(np.uint8)
        sem = sem.cpu().numpy().astype(np.uint8)
        # unravel batch dim
        # bc = list(bc)
        # sem = list(sem)

        # ensure that batch size is one for inference
        assert bc.shape[0] == sem.shape[0] == 1, 'Inference should be run with a batch size of one!'
        bc = bc[0]
        sem = sem[0]

        return {'bc': bc, 'sem': sem}

    def encode_decode(self, img, img_metas, gt_semantic_seg_pre=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if self.tile_inference:
            img, n_h, n_w = split_into_tiles(img, self.inference_tile_size)
            gt_semantic_seg_pre, _, _ = split_into_tiles(gt_semantic_seg_pre, self.inference_tile_size)

        x = self.extract_feat(img)
        output = self._decode_head_forward_test(
            x=x, 
            img_metas=img_metas,
            gt_semantic_seg_pre=gt_semantic_seg_pre
        )
        if self.tile_inference:
            output['bc'] = merge_tiles(output['bc'], n_h, n_w)
            output['sem'] = merge_tiles(output['sem'], n_h, n_w)
        output['bc'] = resize(
            input=output['bc'],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        output['sem'] = resize(
            input=output['sem'],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return output

    def inference(self, img, img_meta, rescale, gt_semantic_seg_pre=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            raise NotImplementedError
            #output = self.slide_inference(img, gt_semantic_seg_pre, img_meta, rescale)
        else:
            output = self.whole_inference(
                img=img, 
                gt_semantic_seg_pre=gt_semantic_seg_pre, 
                img_meta=img_meta, 
                rescale=rescale
            )
        output['bc'] = F.softmax(output['bc'], dim=1)
        output['sem'] = F.softmax(output['sem'], dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output['bc'] = output['bc'].flip(dims=(3, ))
                output['sem'] = output['sem'].flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output['bc'] = output['bc'].flip(dims=(2, ))
                output['sem'] = output['sem'].flip(dims=(2, ))

        return output


    def whole_inference(self, img, img_meta, rescale, gt_semantic_seg_pre=None):
        """Inference with full image."""

        output = self.encode_decode(
            img=img, 
            gt_semantic_seg_pre=gt_semantic_seg_pre, 
            img_metas=img_meta
        )
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            output['bc'] = resize(
                output['bc'],
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False
            )
            output['sem'] = resize(
                output['sem'],
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False
            )

        return output

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, gt_semantic_seg_pre=None, gt_semantic_seg_post=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            inputs=x, 
            img_metas=img_metas, 
            train_cfg=self.train_cfg,
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post
        )

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, gt_semantic_seg_pre=None):
        """Run forward function and calculate loss for decode head in
        inference."""
        output = self.decode_head.forward_test(
            inputs=x, 
            gt_semantic_seg_pre=gt_semantic_seg_pre, 
            img_metas=img_metas, 
            test_cfg=self.test_cfg
        )
        return output

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, gt_semantic_seg_pre=None , gt_semantic_seg_post=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    inputs=x, 
                    img_metas=img_metas,
                    train_cfg=self.train_cfg,
                    gt_semantic_seg_pre=gt_semantic_seg_pre,
                    gt_semantic_seg=gt_semantic_seg,
                    gt_semantic_seg_post=gt_semantic_seg_post
                )
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                inputs=x,
                img_metas=img_metas,
                train_cfg=self.train_cfg,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg=gt_semantic_seg,
                gt_semantic_seg_post=gt_semantic_seg_post  
            )
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

def split_into_tiles(tensor, tile_size):
    if tensor.ndim == 4:
        b, c, h, w = tensor.shape
        assert h % tile_size == 0 and w % tile_size == 0
        n_h = h // tile_size
        n_w = w // tile_size

        tiles = tensor.reshape(b,c,n_h,tile_size,n_w, tile_size)
        tiles = tiles.permute(0,2,4,1,3,5).reshape(b*n_h*n_w, c, tile_size, tile_size)
        return tiles, n_h, n_w
    elif tensor.ndim == 3:
        b, h, w = tensor.shape
        assert h % tile_size == 0 and w % tile_size == 0
        n_h = h // tile_size
        n_w = w // tile_size
        tiles = tensor.reshape(b, n_h, tile_size,n_w, tile_size)
        tiles = tiles.permute(0,1,3,2,4).reshape(b*n_h*n_w, tile_size, tile_size)
        return tiles, n_h, n_w
    else:
        raise ValueError('Invalid number of dimensions: ', tensor.ndim)

def merge_tiles(tiles, n_h, n_w):
    if tiles.ndim == 4:
        n_t, c, t_h, t_w = tiles.shape
        b = n_t // n_h // n_w
        tiles = tiles.reshape(b, n_h, n_w, c, t_h, t_w).permute(0,3,1,4,2,5)
        tensor = tiles.reshape(b, c, n_h * t_h, n_w * t_w)
        return tensor
    if tiles.ndim == 3:
        n_t, t_h, t_w = tiles.shape
        b = n_t // n_h // n_w
        tiles = tiles.reshape(b, n_h, n_w, t_h, t_w).permute(0,1,3,2,4)
        tensor = tiles.reshape(b, n_h * t_h, n_w * t_w)
        return tensor
    else:
        raise ValueError('Invalid number of dimensions: ', tiles.ndim)