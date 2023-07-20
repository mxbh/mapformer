import os.path as osp
import numpy as np

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from ..builder import DATASETS
from ..pipelines import ComposeWithVisualization
from .custom_ccd import CustomDatasetCCD


MEAN = (0,0,0)
STD = (1,1,1)


@DATASETS.register_module()
class HRSCDatasetCCD(CustomDatasetCCD):    
    '''
    HRSCD for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['artificial', 'agricultural', 'forest', 'wetland', 'water']
    
    def __init__(
        self,
        pipeline,
        data_root,
        split,
        ann_dir=None,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        test_mode=False,
        ignore_index_bc=255,
        ignore_index_sem=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        if_visualize=False
    ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.data_root = data_root
        self.img_dir = osp.join(data_root, 'images')
        self.ann_dir = osp.join(data_root, 'labels')
        self.split = split
        with open(osp.join(data_root, 'splits', split + '.txt'), 'r') as f:
            sites = [s.strip() for s in f.readlines()]
        self.sites = sites
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        # load annotations

        self.img_infos = self.load_img_infos()

        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

    def load_img_infos(self):
        img_infos = []
        for site_pre in self.sites:
            d = 'D' + site_pre[:2]

            for tile in mmcv.scandir(osp.join(self.img_dir, '2006', d, site_pre), recursive=False, suffix=self.img_suffix):
                splitted = site_pre.split('-')
                splitted[1] = '2012'
                site_post = '-'.join(splitted)
                tile_seg = tile[:-len(self.img_suffix)] + self.seg_map_suffix
                
                img_info = dict(
                    filename=osp.join(self.img_dir, '2012', d, site_post, tile),
                    filename_pre=osp.join(self.img_dir, '2006', d, site_pre, tile),
                    ann=dict(seg_map=osp.join(self.ann_dir, 'change', d, site_post, tile_seg),
                                seg_map_pre=osp.join(self.ann_dir, '2006', d, site_post, tile_seg), # somehow the files are named with 2012 here as well in the original dataset
                                seg_map_post=osp.join(self.ann_dir, '2012', d, site_post, tile_seg))
                )
                img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} image pairs', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []

        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_bc_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_bc_maps = []
        for img_info in self.img_infos:
            bc_map_file = img_info['ann']['seg_map']
            gt_bc_map = mmcv.imread(
                bc_map_file, flag='unchanged', backend='tifffile')            
            gt_bc_maps.append(gt_bc_map)
        
        return gt_bc_maps

    def get_gt_sem_maps(self, efficient_test=False):
        gt_sem_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info['ann']['seg_map_post']
            gt_seg_map_post = mmcv.imread(
                seg_map_post, flag='unchanged', backend='tifffile')
            # reduce zero label
            # avoid using underflow conversion
            gt_seg_map_post[gt_seg_map_post == 0] = self.ignore_index_sem
            gt_seg_map_post = gt_seg_map_post - 1
            gt_seg_map_post[gt_seg_map_post == self.ignore_index_sem - 1] = self.ignore_index_sem
            gt_sem_maps.append(gt_seg_map_post.astype(np.uint8))

        return gt_sem_maps