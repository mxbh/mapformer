'''
Overall heads for (Conditional) Change Detection, handling both BCD and SCD.
'''
import torch
import torch.nn as nn
from abc import ABCMeta
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, normal_init
from mmcv.utils import build_from_cfg
from .bc_heads import ConcatModule, KConcatModule, ContrastiveModule
from .map_encoders import MAP_ENCODERS
from ..decode_heads.decode_head import BaseDecodeHead
from ..decode_heads import SegformerHead
from .. import builder
from ..builder import HEADS
from ..builder import build_loss
from ..losses import accuracy
from ..cd.fhd import split_batches
from ...core import add_prefix
from ...ops import resize


@HEADS.register_module()
class BaseHeadCCD(nn.Module, metaclass=ABCMeta):
    '''
    Base class with separate subheads for binary change and semantic segmentation.
    '''
    def __init__(self, sem_head, bc_head):
        super(BaseHeadCCD, self).__init__()
        self.sem_head = builder.build_head(sem_head)
        self.bc_head = builder.build_head(bc_head)
        self.inference_wo_sem_info = False

    @property
    def align_corners(self):
        align_corners_in_bc = self.bc_head.align_corners
        align_corners_in_sem = self.sem_head.align_corners

        if align_corners_in_bc and align_corners_in_sem:
            return True
        elif not align_corners_in_bc and not align_corners_in_sem:
            return False
        else: 
            assert False, 'Both subheads should have the same setting for align_corners!'

    @property
    def num_classes(self):
        return self.sem_head.num_classes

    def init_weights(self):
        self.sem_head.init_weights()
        self.bc_head.init_weights()
    
    def forward_train(
        self, 
        inputs,
        img_metas,
        train_cfg, 
        gt_semantic_seg,
        gt_semantic_seg_pre=None,
        gt_semantic_seg_post=None
    ):
        losses_sem = self.sem_head.forward_train( 
            inputs,
            img_metas,
            train_cfg, 
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post
        )
        losses_bc = self.bc_head.forward_train(
            inputs,
            img_metas,
            train_cfg, 
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post
        )
        losses = {}
        losses.update(add_prefix(losses_sem, 'sem'))
        losses.update(add_prefix(losses_bc, 'bc'))

        return losses

    def forward_test(
        self, 
        inputs,  
        img_metas,
        test_cfg,
        gt_semantic_seg_pre=None
    ):
        output_sem = self.sem_head.forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg,
            gt_semantic_seg_pre=gt_semantic_seg_pre
        )
        if self.inference_wo_sem_info:
            b, h, w = gt_semantic_seg_pre.shape
            gt_semantic_seg_pre = self.get_pred_for_semantic_seg_pre(inputs, img_metas, test_cfg)
            gt_semantic_seg_pre = nn.functional.interpolate(
                gt_semantic_seg_pre.unsqueeze(1).float(), 
                size=(h,w)).reshape(b,h,w)

        output_bc = self.bc_head.forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg,
            gt_semantic_seg_pre=gt_semantic_seg_pre
        )
        return {'bc': output_bc, 'sem': output_sem}

    def get_pred_for_semantic_seg_pre(self, inputs, img_metas, test_cfg):
        # reverse inputs
        inputs_reverse = [i.flip(dims=(0,)) for i in inputs]
        output_sem_reverse = self.sem_head.forward_test(
            inputs=inputs_reverse,
            img_metas=img_metas,
            test_cfg=test_cfg,
            gt_semantic_seg_pre=None
        )
        output_sem_pre = output_sem_reverse.flip(dims=(0,))
        return output_sem_pre.argmax(dim=1)


class JointHeadCCD(BaseDecodeHead):
    '''
    Basic head for binary change and semantic segmentation on joint (merged) features.
    '''
    def __init__(
        self,
        loss_decode_bc = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_decode_sem = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index_bc=6,
        ignore_index_sem=6,
        **kwargs
    ):
        super(JointHeadCCD, self).__init__(**kwargs)
        del self.loss_decode
        del self.ignore_index

        self.loss_decode_bc = build_loss(loss_decode_bc)
        self.loss_decode_sem = build_loss(loss_decode_sem)
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem

        self.num_classes_bc = 2
        del self.conv_seg
        self.conv_seg_bc = nn.Conv2d(self.channels, self.num_classes_bc, kernel_size=1)
        self.conv_seg_sem = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index_bc={self.ignore_index_bc}, ' \
            f'ignore_index_sem={self.ignore_index_sem}, ' \
            f'align_corners={self.align_corners}'
        return s

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg_bc, mean=0, std=0.01)
        normal_init(self.conv_seg_sem, mean=0, std=0.01)

    def forward_train(
        self,
        inputs,
        img_metas,
        train_cfg,
        gt_semantic_seg,
        gt_semantic_seg_pre=None,
        gt_semantic_seg_post=None
    ):
        output = self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)
        losses = self.losses(seg_logit_bc=output['bc'], seg_label_bc=gt_semantic_seg, 
                             seg_logit_sem=output['sem'], seg_label_sem=gt_semantic_seg_post)
        return losses


    def forward_test(
        self, 
        inputs,  
        img_metas,
        test_cfg,
        gt_semantic_seg_pre=None
    ):
        return self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit_bc, seg_label_bc, seg_logit_sem, seg_label_sem):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit_bc = resize(
            input=seg_logit_bc,
            size=seg_label_bc.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight_bc = self.sampler.sample(seg_logit_bc, seg_label_bc)
        else:
            seg_weight_bc = None
        seg_label_bc = seg_label_bc.squeeze(1)
        loss['bc.loss_seg'] = self.loss_decode_bc(
            seg_logit_bc,
            seg_label_bc,
            weight=seg_weight_bc,
            ignore_index=self.ignore_index_bc)
        loss['bc.acc_seg'] = accuracy(seg_logit_bc, seg_label_bc)

        seg_logit_sem = resize(
            input=seg_logit_sem,
            size=seg_label_sem.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label_sem = seg_label_sem.squeeze(1)
        seg_weight_sem = None # (seg_label_bc == 1).float()
        loss['sem.loss_seg'] = self.loss_decode_sem(
            seg_logit_sem,
            seg_label_sem,
            weight=seg_weight_sem,
            ignore_index=self.ignore_index_sem)
        loss['sem.acc_seg'] = accuracy(seg_logit_sem, seg_label_sem)
        return loss


    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output_bc = self.conv_seg_bc(feat)
        output_sem = self.conv_seg_sem(feat)
        return {'bc': output_bc, 'sem': output_sem}


@HEADS.register_module()
class JointAttentionHead(JointHeadCCD):
    '''
    Joint binary change and semantic segmentation head based on our attention-style feature fusion.
    '''
    def __init__(self, feature_strides, map_encoder, extra_branch=False, k=None, **kwargs):
        super(JointAttentionHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)
        self.n_semantic_classes = map_encoder['n_semantic_classes']

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index_bc 
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.extra_branch = extra_branch
        if k is None:
            self.k = self.n_semantic_classes + 1
        else:
            self.k = k

        self.temporal_fusion_modules = nn.ModuleList(
            [KConcatModule(
                in_channels=2*self.in_channels[s] + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                k=self.k + (1 if self.extra_branch else 0),
                norm_cfg=self.norm_cfg
            ) for s in range(num_inputs)]
        )
        self.attention_weights = nn.ModuleList(
            [nn.Conv2d(
                in_channels=self.map_encoder.out_channels[s],
                out_channels=self.k * self.channels,
                kernel_size=1,
                ) for s in range(num_inputs)]
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre)
        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f1, f2 = split_batches(x[s])
            m1 = map_features[s]
            if m1.shape[2:] != f1.shape[2:]:
                m1 = resize(m1, size=f1.shape[2:], mode='bilinear', align_corners=self.align_corners)

            h = module(features=[f1, f2, m1])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s](m1) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0], 
                self.k, 
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)

            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out


@HEADS.register_module()
class JointMapFormerHead(JointAttentionHead):
    '''
    Joint binary change and semantic segmentation head for MapFormer.
    '''
    def __init__(
        self,
        feature_strides, 
        map_encoder, 
        extra_branch=False, 
        k=None, 
        contrastive_loss_weight=1.0,
        balance_pos_neg=True,
        **kwargs
    ):
        super(JointMapFormerHead, self).__init__(
            feature_strides=feature_strides,
            map_encoder=map_encoder,
            extra_branch=extra_branch,
            k=k,
            **kwargs
        )
        self.contrastive_img_forward = SegformerHead(
            align_corners = self.align_corners,
            channels=self.channels,
            dropout_ratio=self.dropout_ratio,
            ignore_index=None,
            in_channels=self.in_channels,
            in_index=self.in_index,
            loss_decode={'type': 'CrossEntropyLoss'}, # not used
            norm_cfg=self.norm_cfg,
            num_classes=self.map_encoder.out_channels[0] # embedding dim here
        )
        self.contrastive_module = ContrastiveModule(
            in_channels_map=None, #self.map_encoder.out_channels[0],
            in_channels_img=None,
            proj_channels=self.map_encoder.out_channels[0],
            loss_weight=contrastive_loss_weight,
            balance_pos_neg=balance_pos_neg,
            align_corners=self.align_corners,
        )     

    def forward_train(
        self,
        inputs,
        img_metas,
        train_cfg,
        gt_semantic_seg,
        gt_semantic_seg_pre,
        gt_semantic_seg_post=None
    ):
        #bc_logit = self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)
        #def forward(self, inputs, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre)
        f1_list, f2_list = [], []
        bitemporal_features = []
        contrastive_losses = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f1, f2 = split_batches(x[s])
            m1 = map_features[s]
            if m1.shape[2:] != f1.shape[2:]:
                m1_ = resize(m1, size=f1.shape[2:], mode='bilinear', align_corners=self.align_corners)
            else:
                m1_ = m1

            h = module(features=[f1, f2, m1_])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s](m1_) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0], 
                self.k, 
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)

            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)
            f1_list.append(f1)
            f2_list.append(f2)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        output = self.cls_seg(out)
        losses = self.losses(seg_logit_bc=output['bc'], seg_label_bc=gt_semantic_seg, 
                             seg_logit_sem=output['sem'], seg_label_sem=gt_semantic_seg_post)

        # contrastive loss
        f1_merged = self.contrastive_img_forward(f1_list)
        f2_merged = self.contrastive_img_forward(f2_list)
        contrastive_losses = self.contrastive_module(
            bc=gt_semantic_seg, 
            g1=map_features[0], 
            f2=f2_merged, 
            f1=f1_merged,
            m1=gt_semantic_seg_pre
        )
        losses.update(contrastive_losses)
        return losses


@HEADS.register_module()
class SemSegHeadCCD(nn.Module, metaclass=ABCMeta):
    '''
    Head for change detection via semantic segmentation.
    '''
    def __init__(self, sem_head):
        super(SemSegHeadCCD, self).__init__()
        self.sem_head = builder.build_head(sem_head)
        self.inference_with_gt_sem_pre = False

    @property
    def align_corners(self):
        return self.sem_head.align_corners

    @property
    def num_classes(self):
        return 2

    def init_weights(self):
        self.sem_head.init_weights()
    
    def forward_train(
        self, 
        inputs,
        img_metas,
        train_cfg, 
        gt_semantic_seg,
        gt_semantic_seg_pre=None,
        gt_semantic_seg_post=None
    ):
        losses_sem = self.sem_head.forward_train( 
            inputs,
            img_metas,
            train_cfg, 
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post
        )
        losses = {}
        losses.update(add_prefix(losses_sem, 'sem'))

        return losses

    def forward_test(
        self, 
        inputs,  
        img_metas,
        test_cfg,
        gt_semantic_seg_pre=None
    ):
        output_sem2 = self.sem_head.forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg,
            gt_semantic_seg_pre=None
        )
        if not self.inference_with_gt_sem_pre:
            output_sem1 = self.sem_head.forward_test(
                inputs=[x.flip(dims=(0,)) for x in inputs],
                img_metas=img_metas,
                test_cfg=test_cfg,
                gt_semantic_seg_pre=None
            ).flip(dims=(0,)) # flip two times as only the second half is forwarded by SemHead
            pred_sem1 = output_sem1.argmax(dim=1, keepdim=True)
        else:
            pred_sem1 = resize(gt_semantic_seg_pre.unsqueeze(1).float(), size=output_sem2.shape[-2:], mode='nearest')

        pred_sem2 = output_sem2.argmax(dim=1, keepdim=True)
        pred_bc = 1 - (pred_sem1 == pred_sem2).int()
        output_bc = torch.cat([1. - pred_bc, pred_bc], dim=1).to(output_sem2.dtype)
        return {'bc': output_bc, 'sem': output_sem2}
