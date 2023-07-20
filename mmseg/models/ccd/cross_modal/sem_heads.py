'''
Semantic segmentation subheads for Cross-modal CD.
'''
import torch
from ...builder import HEADS
from ..sem_heads import DummySemHead
from ...decode_heads import SegformerHead

@HEADS.register_module()
class CrossModalSegformerSemHead(SegformerHead):
    '''
    Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        return super(CrossModalSegformerSemHead, self).forward_train(
            inputs=inputs, 
            img_metas=img_metas, 
            train_cfg=train_cfg, 
            gt_semantic_seg=gt_semantic_seg_post
        )

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHead, self).forward_test(
            inputs=inputs, 
            img_metas=img_metas, 
            test_cfg=test_cfg)


@HEADS.register_module()
class CrossModalDummySemHead(DummySemHead):
    '''
    Placeholder for semantic segmentation head if one is only interested in BCD (not SCD).
    '''
    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        B, _, H, W = inputs[0].shape
        return torch.rand(B, self.num_classes, H, W, dtype=inputs[0].dtype, device=inputs[0].device)