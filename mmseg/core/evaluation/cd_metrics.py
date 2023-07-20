import torch
from torch import Tensor
import numpy as np
from collections import OrderedDict
from .metrics import eval_metrics, total_intersect_and_union, f_score


def cd_eval_metrics(results,
                    gt_seg_maps,
                    num_classes,
                    ignore_index,
                    metrics=['mIoU'],
                    nan_to_num=None,
                    label_map=dict(),
                    reduce_zero_label=False,
                    beta=1,
                    ):
    my_metrics = ['mFscoreCD']
    original_allowed_metrics = ['mIoU', 'mDice', 'mFscore', ]
    allowed_metrics = my_metrics + original_allowed_metrics

    if isinstance(metrics, str):
        metrics = [metrics]

    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = OrderedDict()
    for metric in metrics:
        if set([metric]).issubset(set(original_allowed_metrics)):
            ret_metrics.update(eval_metrics(results,
                                            gt_seg_maps,
                                            num_classes,
                                            ignore_index,
                                            metric,
                                            nan_to_num,
                                            label_map,
                                            reduce_zero_label,
                                            beta))
        elif set([metric]).issubset(set(my_metrics)):
            total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results, gt_seg_maps, num_classes, ignore_index, label_map,
                reduce_zero_label)

            all_acc = total_area_intersect.sum() / total_area_label.sum()
            ret_metrics.update(OrderedDict({'aAcc': all_acc}))

            if metric == 'mFscoreCD':
                total_area_intersect_sum = total_area_intersect[1:].sum()
                total_area_pred_label_sum = total_area_pred_label[1:].sum()
                total_area_label_sum = total_area_label[1:].sum()
           
                precision = total_area_intersect_sum / total_area_pred_label_sum
                recall = total_area_intersect_sum / total_area_label_sum
                f_value = torch.tensor([f_score(precision, recall, beta)])
                ret_metrics['FscoreCD'] = f_value
                ret_metrics['PrecisionCD'] = precision
                ret_metrics['RecallCD'] = recall

    for metric, value in ret_metrics.copy().items():
        if isinstance(value, Tensor):
            ret_metrics.update({metric: value.numpy()})

    if nan_to_num is not None:
        ret_metrics.update(OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        }))

    return ret_metrics
