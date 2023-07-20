'''
Map encoders.
'''
import torch
from torch import nn
from mmcv.cnn import ConvModule
from ...ops import resize

from mmcv.utils import Registry

MAP_ENCODERS = Registry('map_encoder')


@MAP_ENCODERS.register_module()
class BasicMapEncoder(nn.Module):
    '''
    Three-layer CNN for encoding map info.
    '''
    def __init__(
        self, 
        n_semantic_classes, 
        ignore_index, 
        out_channels, 
        scale, 
        num_scales, 
        norm_cfg
    ):
        super(BasicMapEncoder, self).__init__()
        self.n_semantic_classes = n_semantic_classes
        self.ignore_index = ignore_index
        self.num_scales = num_scales
        self.scale = [scale for _ in range(self.num_scales)]
        self.out_channels = [out_channels for _ in range(self.num_scales)]

        self.layer1 = nn.Conv2d(
            in_channels=n_semantic_classes + 1,  # one extra for ignore index
            out_channels=self.out_channels[0],
            kernel_size=1,
        )
        self.layer2 = ConvModule(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg
        )
        self.layer3 = ConvModule(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg
        )

    def init_weights(self):
        pass

    def forward(self, gt_semantic_seg_pre):
        if gt_semantic_seg_pre.ndim == 4:
            gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)
        B, H, W = gt_semantic_seg_pre.shape
        one_hot_channels = self.n_semantic_classes + 1
        # last index for ignore
        if self.ignore_index > self.n_semantic_classes:
            _gt_semantic_seg_pre = torch.clone(gt_semantic_seg_pre)
            _gt_semantic_seg_pre[gt_semantic_seg_pre == self.ignore_index] = self.n_semantic_classes
        else:
            _gt_semantic_seg_pre = gt_semantic_seg_pre
        with torch.no_grad():
            one_hot = nn.functional.one_hot(
                _gt_semantic_seg_pre.long(), num_classes=one_hot_channels)
            one_hot = one_hot.permute(0, 3, 1, 2).reshape(
                B, one_hot_channels, H, W).float()
            one_hot = resize(one_hot, scale_factor=self.scale[0],
                             mode='bilinear', align_corners=False)

        x = self.layer1(one_hot)
        x = self.layer2(x)
        x = self.layer3(x)
        return [x for _ in range(self.num_scales)]


@MAP_ENCODERS.register_module()
class HighLevelMapEncoder(BasicMapEncoder):
    '''
    Map encoder with consolidated classes.
    '''
    def __init__(
        self,  
        ignore_index, 
        out_channels, 
        scale, 
        num_scales, 
        norm_cfg,
        label_map, # list of list of indices, e.g. [[1,2,3], [4,5,6]] 
        n_semantic_classes=None # unused, only for compatibility
    ):
        super(HighLevelMapEncoder, self).__init__(
            n_semantic_classes=len(label_map) - 1, # -1 because ignore index is added in the BaseEncoder
            ignore_index=ignore_index,
            out_channels=out_channels,
            scale=scale,
            num_scales=num_scales,
            norm_cfg=norm_cfg    
        )
        self.label_map = label_map


    def init_weights(self):
        pass

    def forward(self, gt_semantic_seg_pre):
        if gt_semantic_seg_pre.ndim == 4:
            gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)

        gt_semantic_seg_pre_mapped = []
        for lm in self.label_map:
            mapped = sum([(gt_semantic_seg_pre == l) for l in lm]).float()
            # result should be zeros and ones
            assert set(mapped.unique().tolist()).issubset({0.,1.})
            gt_semantic_seg_pre_mapped.append(mapped)

        one_hot = torch.stack(gt_semantic_seg_pre_mapped, dim=1)

        with torch.no_grad():
            one_hot = resize(one_hot, scale_factor=self.scale[0],
                             mode='bilinear', align_corners=False)

        x = self.layer1(one_hot)
        x = self.layer2(x)
        x = self.layer3(x)
        return [x for _ in range(self.num_scales)]


@MAP_ENCODERS.register_module()
class LowResMapEncoder(BasicMapEncoder):
    '''
    Map encoder with downsampling to mimic low res map info.
    '''
    def __init__(self, resolution_factor, *args, **kwargs):
        super(LowResMapEncoder, self).__init__(*args, **kwargs)
        self.resolution_factor = resolution_factor

    def forward(self, gt_semantic_seg_pre):
        if gt_semantic_seg_pre.ndim == 3:
            gt_semantic_seg_pre = gt_semantic_seg_pre.unsqueeze(1)
        with torch.no_grad():
            gt_semantic_seg_pre_low = nn.functional.interpolate(
                input=gt_semantic_seg_pre.float(),
                scale_factor=1 / self.resolution_factor,
                mode='nearest'
            )
            gt_semantic_seg_pre_low = nn.functional.interpolate(
                input=gt_semantic_seg_pre_low,
                size=gt_semantic_seg_pre.shape[-2:],
                mode='nearest'
            )
        return super(LowResMapEncoder, self).forward(gt_semantic_seg_pre_low)