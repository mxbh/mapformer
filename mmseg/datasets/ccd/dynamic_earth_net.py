import os.path as osp

import mmcv
from mmcv.utils import print_log
from datetime import datetime

from mmseg.utils import get_root_logger
from ..builder import DATASETS
from ..pipelines import ComposeWithVisualization
from ..custom import CustomDataset
from .custom_ccd import CustomDatasetCCD


MEAN = (649.9822,  862.5364,  939.1118, 2521.4004)
STD = (654.9196,  727.9036,  872.8431, 1035.1437)

@DATASETS.register_module()
class DynamicEarthNet(CustomDataset):
    '''
    Base class for DynamicEarthNet.
    '''
    def __init__(self,
                 pipeline,
                 data_root,
                 split,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 test_mode=False,
                 ignore_index=6,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 pair_mode='consecutive', # 'all'
                 if_visualize=False,
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
        self.observations = self.load_observations()
        assert pair_mode in ['consecutive', 'all'], f'Unknown pair mode "{pair_mode}"!' 
        self.pair_mode = pair_mode
        self.img_infos = self.generate_pairs()

        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

    def load_observations(self):
        observations = {}

        for site in self.sites:
            observations[site] = []
            for seg_file in mmcv.scandir(osp.join(self.ann_dir, site), self.img_suffix, recursive=False):
                date = datetime.strptime(seg_file[:-len(self.seg_map_suffix)], '%Y-%m-%d')
                observations[site].append(date)

        # print_log(f'Loaded {len(observations) * 24} images', logger=get_root_logger())
        return observations
    
    def generate_pairs(self):
        img_infos = []
        for site in self.sites:
            dates = sorted(self.observations[site])
            date_strings = [date.strftime('%Y-%m-%d') for date in dates]
            if self.pair_mode == 'consecutive':
                for i in range(len(dates) - 1):
                    img_info = dict(filename=osp.join(self.img_dir, site, date_strings[i+1]+self.img_suffix),
                                    filename_pre=osp.join(self.img_dir, site, date_strings[i]+self.img_suffix),
                                    ann=dict(seg_map=osp.join(self.ann_dir, site, date_strings[i+1]+self.seg_map_suffix),
                                             seg_map_pre=osp.join(self.ann_dir, site, date_strings[i]+self.seg_map_suffix)
                                    )
                    )
                    img_infos.append(img_info)
            elif self.pair_mode == 'all':
                for i in range(len(dates) - 1):
                    for j in range(i+1, len(dates)):
                        img_info = dict(filename=osp.join(self.img_dir, site, date_strings[j]+self.img_suffix),
                                        filename_pre=osp.join(self.img_dir, site, date_strings[i]+self.img_suffix),
                                        ann=dict(seg_map=osp.join(self.ann_dir, site, date_strings[j]+self.seg_map_suffix),
                                                 seg_map_pre=osp.join(self.ann_dir, site, date_strings[i]+self.seg_map_suffix)

                                        )
                        )
                        img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} image pairs', logger=get_root_logger())
        return img_infos
        

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        # results['img_prefix'] = ''
        # results['img_pre_prefix'] = ''
        # results['seg_prefix'] = ''
        # results['seg_pre_prefix'] = ''
        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

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


@DATASETS.register_module()
class DynamicEarthNetCCD(DynamicEarthNet, CustomDatasetCCD):
    '''
    DynamicEarthNet for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['impervious', 'agriculture', 'vegetation', 'wetlands', 'soil', 'water']
    
    def __init__(
        self,
        pipeline,
        data_root,
        split,
        ann_dir=None,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        test_mode=False,
        ignore_index_bc=6,
        ignore_index_sem=6,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        pair_mode='consecutive', # 'all'
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
        self.observations = self.load_observations()
        assert pair_mode in ['consecutive', 'all'], f'Unknown pair mode "{pair_mode}"!' 
        self.pair_mode = pair_mode
        self.img_infos = self.generate_pairs()

        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

    def get_gt_bc_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_bc_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info['ann']['seg_map']
            seg_map_pre = img_info['ann']['seg_map_pre']
            gt_seg_map_pre = mmcv.imread(
                seg_map_pre, flag='unchanged', backend='tifffile').argmax(axis=2)
            gt_seg_map_post = mmcv.imread(
                seg_map_post, flag='unchanged', backend='tifffile').argmax(axis=2)
            gt_bc_map = (gt_seg_map_pre != gt_seg_map_post).astype(gt_seg_map_post.dtype)

            # get rid of ignore class
            gt_bc_map[gt_seg_map_pre == self.ignore_index_sem] = self.ignore_index_bc
            gt_bc_map[gt_seg_map_post == self.ignore_index_sem] = self.ignore_index_bc
            
            gt_bc_maps.append(gt_bc_map)
        
        return gt_bc_maps

    def get_gt_sem_maps(self, efficient_test=False):
        gt_sem_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info['ann']['seg_map']
            gt_seg_map_post = mmcv.imread(
                seg_map_post, flag='unchanged', backend='tifffile').argmax(axis=2)
            gt_sem_maps.append(gt_seg_map_post)

        return gt_sem_maps

            