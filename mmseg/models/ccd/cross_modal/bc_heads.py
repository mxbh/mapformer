'''
Binary change subheads for Cross-modal CD.
'''
import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmcv.cnn import ConvModule

from mmseg.models.cd.fhd import split_batches
from mmseg.models.decode_heads import SegformerHead
from ..bc_heads import BaseHeadBC, ConcatModule, KConcatModule, ContrastiveModule
from ..map_encoders import MAP_ENCODERS
from ...builder import HEADS
from ....ops import resize

@HEADS.register_module()
class CrossModalConcathead(BaseHeadBC):
    '''
    BC head for cross-modal concatentation baseline.
    '''
    def __init__(self, feature_strides, map_encoder, **kwargs):
        super(CrossModalConcathead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.temporal_fusion_modules = nn.ModuleList(
            [ConcatModule(
                in_channels=self.in_channels[s] + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                norm_cfg=self.norm_cfg
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
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1 = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)

            f = module(features=[f2, m1])
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out

@HEADS.register_module()
class CrossModalAttentionHead(BaseHeadBC):
    '''
    BC subhead for cross-modal CD based on our attention-style feature fusion.
    '''
    def __init__(self, feature_strides, map_encoder, extra_branch=False, k=None, **kwargs):
        '''
        :param k: How many channels each feature can attend to.
        '''
        super(CrossModalAttentionHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)
        self.n_semantic_classes = map_encoder['n_semantic_classes']

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index # TODO: this look at the wrong ignore index (bc instead of sem), but doesn't matter as long as the two indices are equal
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.extra_branch = extra_branch
        if k is None:
            self.k = self.n_semantic_classes + 1
        else:
            self.k = k

        self.temporal_fusion_modules = nn.ModuleList(
            [KConcatModule(
                in_channels=self.in_channels[s] + self.map_encoder.out_channels[s],
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
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1 = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)

            h = module(features=[f2, m1])

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
class CrossModalMapFormerHead(CrossModalAttentionHead):
    '''
    BC subhead for cross-modal MapFormer.
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
        super(CrossModalMapFormerHead, self).__init__(
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
            align_corners=self.align_corners
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
        f2_list = []
        bitemporal_features = []
        contrastive_losses = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1_ = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)
            else:
                m1_ = m1

            h = module(features=[f2, m1_])

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
            f2_list.append(f2)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        bc_logit = self.cls_seg(out)
        losses = self.losses(seg_logit=bc_logit, seg_label=gt_semantic_seg)

        f2_merged = self.contrastive_img_forward(f2_list)
        contrastive_losses = self.contrastive_module(
            bc=gt_semantic_seg, 
            g1=map_features[0], 
            f2=f2_merged, 
            f1=None
        )
        losses.update(contrastive_losses)
        return losses