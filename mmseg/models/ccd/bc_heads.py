'''
Subheads for binary change detection.
'''
import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmcv.cnn import ConvModule

from mmseg.models.cd.fhd import split_batches
from mmseg.models.decode_heads import SegformerHead
from .map_encoders import MAP_ENCODERS
from ..builder import HEADS
from ..decode_heads.decode_head import BaseDecodeHead
from ..cd.fhd import MLP, FHD_Module
from ...ops import resize


class BaseHeadBC(BaseDecodeHead):
    '''
    Base class for binary change subheads.
    '''
    def forward_train(
        self,
        inputs,
        img_metas,
        train_cfg,
        gt_semantic_seg,
        gt_semantic_seg_pre=None,
        gt_semantic_seg_post=None
    ):
        bc_logit = self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)
        losses = self.losses(seg_logit=bc_logit, seg_label=gt_semantic_seg)
        return losses

    def forward_test(
        self,
        inputs,
        img_metas,
        test_cfg,
        gt_semantic_seg_pre=None
    ):
        return self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)


class ConcatModule(nn.Module):
    '''
    Module for concatenation multiple features along channel dim.
    '''
    def __init__(self, in_channels, out_channels, norm_cfg):
        super(ConcatModule, self).__init__()
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)

    def forward(self, features, weight_inputs=None):
        '''
        Assumes features = [f1,f2,m1] or features = [f1,f2] or already concatenated tensors.
        '''
        concat_features = features if isinstance(features, torch.Tensor) else torch.cat(features, dim=1)
        f = self.conv1(concat_features)
        return self.conv2(f)


class KConcatModule(nn.Module):
    '''
    Module for extracting K representations of joint features in parallel.
    '''
    def __init__(self, in_channels, out_channels, k, norm_cfg):
        super(KConcatModule, self).__init__()
        self.conv1 = ConvModule(in_channels, out_channels * k, kernel_size=1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(out_channels * k, out_channels * k, kernel_size=1, norm_cfg=norm_cfg,
                                groups=k)

    def forward(self, features, weight_inputs=None):
        '''
        Assumes features = [f1,f2,m1] or features = [f1,f2] or already concatenated tensors.
        '''
        concat_features = features if isinstance(features, torch.Tensor) else torch.cat(features, dim=1)
        f = self.conv1(concat_features)
        return self.conv2(f)


class ContrastiveModule(nn.Module):
    '''
    Contrastive loss module.
    '''
    def __init__(
            self, 
            in_channels_map, 
            in_channels_img, 
            proj_channels,
            loss_weight, 
            balance_pos_neg,
            align_corners,
            margin=0
        ):
        super(ContrastiveModule, self).__init__()
        if in_channels_map is not None:
            self.map_proj = nn.Conv2d(in_channels_map, proj_channels, kernel_size=1)
        else:
            self.map_proj = lambda x: x.detach()
        if in_channels_img is not None:
            self.img_proj = nn.Conv2d(in_channels_img, proj_channels, kernel_size=1)
        else:
            self.img_proj = nn.Identity()
        self.loss_weight = loss_weight
        self.balance_pos_neg = balance_pos_neg
        self.margin = margin
        self.align_corners = align_corners

    def forward(self, bc, g1, f2, f1=None, m1=None):
        B = g1.shape[0]
        H_g, W_g = g1.shape[-2:]
        H_f, W_f = f2.shape[-2:]
        H, W = min(H_g, H_f), min(W_g, W_f)

        g1 = resize(input=g1, size=(H,W), mode='bilinear', align_corners=self.align_corners)
        g1_proj = self.map_proj(g1)

        f2 = resize(input=f2, size=(H,W), mode='bilinear', align_corners=self.align_corners)
        f2_proj = self.img_proj(f2)
        bc = resize(input=bc.float(), size=(H,W), mode='nearest')
        mask = (bc == 1).flatten() 
        cos2 = nn.functional.cosine_similarity(g1_proj, f2_proj, dim=1).flatten()

        if f1 is not None:
            f1 = resize(input=f1, size=(H,W), mode='bilinear', align_corners=self.align_corners)
            f1_proj = self.img_proj(f1)
            cos1 = nn.functional.cosine_similarity(g1_proj, f1_proj, dim=1).flatten()

        N_pixel = B * H * W * 2 if f1 is not None else B * H * W
        N_neg = mask.sum()
        w_pos = N_pixel if not self.balance_pos_neg else N_pixel - N_neg + torch.finfo(torch.float).eps
        w_neg = N_pixel if not self.balance_pos_neg else N_neg + torch.finfo(torch.float).eps

        loss2_pos = -cos2[~mask].sum() / w_pos
        loss2_neg = torch.maximum(cos2[mask] - self.margin, torch.tensor(0., device=cos2.device)).sum() / w_neg
        loss2_neg = loss2_neg + self.margin
        
        if f1 is not None:
            loss1_pos = -cos1.sum() / w_pos
        else:
            loss1_pos = torch.tensor(0.)

        contrastive_losses = {
            'contrastive_loss.t1_pos': loss1_pos * self.loss_weight,
            'contrastive_loss.t2_pos': loss2_pos * self.loss_weight,
            'contrastive_loss.t2_neg': loss2_neg * self.loss_weight
        }  
        return contrastive_losses


class DynamicMLP_C(nn.Module):
    '''
    Dynamic MLP version C.
    Adapted from https://github.com/ylingfeng/DynamicMLP/blob/773197c5a0aaf15a4d4c8b54a4765ea41f784d3f/models/dynamic_mlp.py#L124
    '''
    def __init__(
        self, 
        in_channels_features, 
        in_channels_weights, 
        out_channels, 
        channel_factor,
    ):
        super(DynamicMLP_C, self).__init__()
        self.in_channels_features = in_channels_features
        self.in_channels_weights = in_channels_weights
        self.out_channels = out_channels
        self.channel_factor = channel_factor
        self.down_channels = in_channels_features // channel_factor

        self.conv11 = nn.Sequential(
            nn.Conv2d(self.in_channels_features, self.out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(self.out_channels, self.down_channels, kernel_size=1)

        self.conv21 = nn.Sequential(
            nn.Conv2d(self.in_channels_weights, self.out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(self.out_channels, self.down_channels**2, kernel_size=1)

        self.br = nn.Sequential(
            nn.LayerNorm(self.down_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(self.down_channels, self.out_channels, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(self.out_channels)
        self.relu = nn.ReLU()
        

    def forward(self, features, weight_inputs):
        '''
        Assumes list of features maps of the same size for both features and weight_inputs.
        E.g. features=[f1,f2] and weight_inputs=[f1,f2,m1]
        '''
        cat_fea_f = torch.cat(features, dim=1)
        cat_fea_w = torch.cat(weight_inputs, dim=1)

        weight11 = self.conv11(cat_fea_f)
        weight12 = self.conv12(weight11) # (B,C,H,W)

        weight21 = self.conv21(cat_fea_w)
        weight22 = self.conv22(weight21) # (B,C**2,H,W)

        B, C, H, W = weight12.shape
        img_fea = torch.bmm(
            weight12.permute(0,2,3,1).reshape(B * H * W, 1, C),
            weight22.permute(0,2,3,1).reshape(B * H * W, C, C)
        ).reshape(B, H, W, C)
        img_fea = self.br(img_fea)
        img_fea = img_fea.permute(0,3,1,2) # (B,C,H,W)
        img_fea = self.conv3(img_fea)
        img_fea = self.ln(img_fea.permute(0,2,3,1)).permute(0,3,1,2)
        img_fea = self.relu(img_fea)

        return img_fea


@HEADS.register_module()
class ConcatHead(BaseHeadBC):
    '''
    BC head for bi-temporal concatenation baseline.
    '''
    def __init__(self, feature_strides, map_encoder=None, **kwargs):
        super(ConcatHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)

        self.temporal_fusion_modules = nn.ModuleList(
            [ConcatModule(
                in_channels=2*self.in_channels[s],
                out_channels=self.channels,
                norm_cfg=self.norm_cfg
            ) for s in range(num_inputs)]
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, gt_semantic_seg_pre=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16

        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f1, f2 = split_batches(x[s])
            f = module(features=[f1, f2])
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out


@HEADS.register_module()
class CondConcathead(BaseHeadBC):
    '''
    BC head for condition change detection concatenation baseline.
    '''
    def __init__(self, feature_strides, map_encoder, **kwargs):
        super(CondConcathead, self).__init__(input_transform='multiple_select', **kwargs)
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
                in_channels=2*self.in_channels[s] + self.map_encoder.out_channels[s],
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
            f1, f2 = split_batches(x[s])
            m1 = map_features[s]
            if m1.shape[2:] != f1.shape[2:]:
                m1 = resize(m1, size=f1.shape[2:], mode='bilinear', align_corners=self.align_corners)

            f = module(features=[f1, f2, m1])
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out
        

@HEADS.register_module()
class DynamicMLPHead(BaseHeadBC):
    '''
    BC head for DynamicMLP baseline.
    '''
    def __init__(
        self,
        feature_strides, 
        map_encoder, 
        channel_factor, 
        weight_inputs='all', # map 
        **kwargs
    ):
        super(DynamicMLPHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        assert weight_inputs in ['all', 'map']
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)
        self.channel_factor = channel_factor

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.weight_inputs = weight_inputs
        self.temporal_fusion_modules = nn.ModuleList(
            [DynamicMLP_C(
                in_channels_features=2*self.in_channels[s] + self.map_encoder.out_channels[s],
                in_channels_weights=(2*self.in_channels[s] if self.weight_inputs == 'all' else 0) \
                    + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                channel_factor=self.channel_factor,
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

            weight_inputs = [f1, f2, m1] if self.weight_inputs == 'all' else [m1]
            f = module(features=[f1, f2, m1], weight_inputs=weight_inputs)
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out


@HEADS.register_module()
class FHD_Head(BaseHeadBC):
    '''
    BC head for FHD.
    Taken from https://github.com/ZSVOS/FHD/blob/f1f55a1912b0488ae8dd701bdba8584f63ef169c/mmseg/models/cd/fhd.py#L42.
    '''
    def __init__(self, feature_strides, **kwargs):
        super(FHD_Head, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.channels)

        self.linear_fuse = ConvModule(
            in_channels=self.channels * 4,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.fhd1 = FHD_Module(channels=c1_in_channels, r=c1_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd2 = FHD_Module(channels=c2_in_channels, r=c2_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd3 = FHD_Module(channels=c3_in_channels, r=c3_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd4 = FHD_Module(channels=c4_in_channels, r=c4_in_channels // 16, norm_cfg=self.norm_cfg)

    def forward(self, inputs, gt_semantic_seg_pre=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        c1, c2, c3, c4 = x
        # bm: backworad image; fm: forward image
        bm1, fm1 = split_batches(c1)
        bm2, fm2 = split_batches(c2)
        bm3, fm3 = split_batches(c3)
        bm4, fm4 = split_batches(c4)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = bm4.shape

        d1 = self.fhd1(bm1, fm1)
        d2 = self.fhd2(bm2, fm2)
        d3 = self.fhd3(bm3, fm3)
        d4 = self.fhd4(bm4, fm4)
        # Diff: 4
        _d4 = self.linear_c4(d4).permute(0, 2, 1).reshape(n, -1, d4.shape[2], d4.shape[3])
        _d4 = resize(_d4, size=bm1.size()[2:], mode='bilinear', align_corners=False)
        # Diff: 3
        _d3 = self.linear_c3(d3).permute(0, 2, 1).reshape(n, -1, d3.shape[2], d3.shape[3])
        _d3 = resize(_d3, size=bm1.size()[2:], mode='bilinear', align_corners=False)
        # Diff: 2
        _d2 = self.linear_c2(d2).permute(0, 2, 1).reshape(n, -1, d2.shape[2], d2.shape[3])
        _d2 = resize(_d2, size=bm1.size()[2:], mode='bilinear', align_corners=False)
        # Diff: 1
        _d1 = self.linear_c1(d1).permute(0, 2, 1).reshape(n, -1, d1.shape[2], d1.shape[3])
        # prediction
        _d = self.linear_fuse(torch.cat([_d4, _d3, _d2, _d1], dim=1))
        #d = self.dropout(_d)
        d = self.cls_seg(_d)
        return d


@HEADS.register_module()
class AttentionHead(BaseHeadBC):
    '''
    BC head based on our attention-style feature fusion.
    '''
    def __init__(
        self, 
        feature_strides, 
        map_encoder, 
        extra_branch=False, 
        k=None, 
        **kwargs
    ):
        '''
        :param k: How many channels each feature can attend to.
        '''
        super(AttentionHead, self).__init__(input_transform='multiple_select', **kwargs)
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
class MapFormerHead(AttentionHead):
    '''
    BC for MapFormer.
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
        super(MapFormerHead, self).__init__(
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
        bc_logit = self.cls_seg(out)
        losses = self.losses(seg_logit=bc_logit, seg_label=gt_semantic_seg)

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