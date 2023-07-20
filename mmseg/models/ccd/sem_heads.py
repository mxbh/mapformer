'''
Subheads for semantic segmentation.
'''
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.ops import resize
from mmcv.runner import force_fp32
from ..losses import accuracy
from ..cd.fhd import split_batches
from ..decode_heads import SegformerHead
from ..builder import HEADS


@HEADS.register_module()
class SegformerSemHead(SegformerHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class DummySemHead(nn.Module):
    '''
    Placeholder for semantic segmentation head if one is only interested in BCD (not SCD).
    '''
    def __init__(self, num_classes, align_corners, **kwargs):
        super(DummySemHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners

    def init_weights(self, *args, **kwargs):
        pass
    
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        losses = dict(
            loss_seg = torch.tensor(0., dtype=inputs[0].dtype, device=inputs[0].device),
            acc_seg = torch.tensor(0., dtype=inputs[0].dtype, device=inputs[0].device)
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        Bx2, _, H, W = inputs[0].shape
        return torch.rand(Bx2 // 2, self.num_classes, H, W, dtype=inputs[0].dtype, device=inputs[0].device)
