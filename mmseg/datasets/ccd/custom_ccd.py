import os.path as osp
from functools import reduce

import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import scd_eval_metrics
from ..builder import DATASETS
from ..custom_cd import CustomDatasetCD
from ..pipelines import ComposeWithVisualization


@DATASETS.register_module()
class CustomDatasetCCD(CustomDatasetCD):
    '''
    Base class for datasets for Conditional CD.
    '''
    def __init__(self,
                 pipeline,
                 img1_dir,
                 img2_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index_bc=255,
                 ignore_index_sem=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 if_visualize=False,
                 ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img1_dir):
                self.img1_dir = osp.join(self.data_root, self.img1_dir)
                self.img2_dir = osp.join(self.data_root, self.img2_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img1_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric: Dummy argument for compatibility.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        gt_bc_maps = self.get_gt_bc_maps(efficient_test)
        gt_sem_maps = self.get_gt_sem_maps(efficient_test)

        if self.CLASSES is None:
            num_semantic_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_sem_maps]))
        else:
            num_semantic_classes = len(self.CLASSES)

        ret_metrics = scd_eval_metrics(
            results=results,
            gt_bc_maps=gt_bc_maps,
            gt_sem_maps=gt_sem_maps,
            num_semantic_classes=num_semantic_classes,
            ignore_index_bc=self.ignore_index_bc,
            ignore_index_sem=self.ignore_index_sem
        )

        if self.CLASSES is None:
            class_names = tuple(range(num_semantic_classes))
        else:
            class_names = self.CLASSES

        SCD_metrics = ['BC', 'BC_precision', 'BC_recall', 'SC', 'SCS', 'mIoU']
        summary_table = PrettyTable(field_names=SCD_metrics)
        summary_table.add_row([np.round(ret_metrics[m], decimals=3) for m in SCD_metrics])

        print_log('Summary:', logger=logger)
        print_log('\n' + summary_table.get_string(), logger=logger)

        classwise_table = PrettyTable(field_names=['Class'] + list(class_names))
        classwise_table.add_row(['IoU'] + list(np.round(ret_metrics['IoU_per_class'], decimals=3)))
        classwise_table.add_row(['SC'] + list(np.round(ret_metrics['SC_per_class'], decimals=3)))

        print_log('per class results:', logger=logger)
        print_log('\n' + classwise_table.get_string(), logger=logger)

        return ret_metrics