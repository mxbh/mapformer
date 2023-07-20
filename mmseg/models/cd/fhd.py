import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F

from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer, kaiming_init)
from mmcv.runner import Sequential

from mmseg.utils import split_images
from mmseg.ops import resize
from .. import builder
from ..builder import BACKBONES
from ..builder import HEADS
from ..decode_heads.decode_head import BaseDecodeHead


@BACKBONES.register_module()
class BitemporalBackbone(nn.Module):
    def __init__(self, backbone_method='mit', **kargs):
        assert isinstance(backbone_method, str), \
            f'merge_method should a str object, but got {type(backbone_method)}'
        self.backbone_method = backbone_method
        super().__init__()
        kargs.update(type=kargs.pop('ori_type'))
        self.backbone = builder.build_backbone(kargs)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def forward(self, x):
        x1, x2 = split_images(x)
        x = merge_batches(x1, x2)
        x = self.backbone(x)

        return x


#@HEADS.register_module()
class FHD_Head(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(FHD_Head, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.fhd1 = FHD_Module(channels=c1_in_channels, r=c1_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd2 = FHD_Module(channels=c2_in_channels, r=c2_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd3 = FHD_Module(channels=c3_in_channels, r=c3_in_channels // 16, norm_cfg=self.norm_cfg)
        self.fhd4 = FHD_Module(channels=c4_in_channels, r=c4_in_channels // 16, norm_cfg=self.norm_cfg)

    def forward(self, inputs):
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
        d = self.dropout(_d)
        d = self.linear_pred(d)
        return d


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# (TSC) Time-Specific Context
class TSC(nn.Module):
    def __init__(self, scale):
        super(TSC, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        # b, c, h, w = feats.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


# (TSA) Time-Specific Aggregation
class TSA(_SelfAttentionBlock):
    def __init__(self, channels, inter_channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(TSA, self).__init__(
            key_in_channels=channels,
            query_in_channels=channels,
            channels=inter_channels,
            out_channels=channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            channels * 2,
            channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(TSA,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


# FHD (Feature Hierarchical Differentiation)
class FHD_Module(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 # norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(FHD_Module, self).__init__()
        inter_channels = int(channels // r)
        
        #-------------------------------------   HD   -------------------------------------#
        # local attention
        la_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        la_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        la_act1 = build_activation_layer(act_cfg)
        la_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        la_bn2 = build_norm_layer(norm_cfg, channels)[1]
        la_layers = [la_conv1, la_bn1, la_act1, la_conv2, la_bn2]
        self.la_layers = Sequential(*la_layers)
        # globla attention
        aap = nn.AdaptiveAvgPool2d(1)
        ga_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        ga_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        ga_act1 = build_activation_layer(act_cfg)
        ga_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        ga_bn2 = build_norm_layer(norm_cfg, channels)[1]
        ga_layers = [aap, ga_conv1, ga_bn1, ga_act1, ga_conv2, ga_bn2]
        self.ga_layers = Sequential(*ga_layers)

        self.sigmoid = nn.Sigmoid()
        #----------------------------------------------------------------------------------#
        
        #-------------------------------------   TSA   ------------------------------------#
        self.tsa_bm = TSA(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        
        self.tsa_fm = TSA(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        #----------------------------------------------------------------------------------#
        
        #-------------------------------------   TSC   ------------------------------------#
        self.tsc_bm = TSC(scale=1)
        self.tsc_fm = TSC(scale=1)
        #----------------------------------------------------------------------------------#
        
        #-------------------------------------   TSM   ------------------------------------# 
        self.tsm_bm = ConvModule(channels, 2, kernel_size=1)
        self.tsm_fm = ConvModule(channels, 2, kernel_size=1)
        #----------------------------------------------------------------------------------#

    def forward(self, bm, fm):
        bm_pred = self.tsm_bm(bm)
        fm_pred = self.tsm_fm(fm)
        bm_context = self.tsc_bm(bm, bm_pred)
        fm_context = self.tsc_fm(fm, fm_pred)
        bm_agg = self.tsa_bm(bm, bm_context)
        fm_agg = self.tsa_fm(fm, fm_context)

        agg = bm_agg + fm_agg
        agg_loc = self.la_layers(agg)
        agg_glo = self.ga_layers(agg)
        agg_lg = agg_loc + agg_glo
        w = self.sigmoid(agg_lg)

        diff = 2 * bm_agg * w + 2 * fm_agg * (1 - w)
        return diff


def split_batches(x: Tensor):
    """ Split a 2*B batch of images into two B images per batch,
    in order to adapt to MMSegmentation """

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size:, ...]
    return x1, x2


def merge_batches(x1: Tensor, x2: Tensor):
    """ merge two batches each contains B images into a 2*B batch of images
    in order to adapt to MMSegmentation """

    assert x1.ndim == 4 and x2.ndim == 4, f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)
