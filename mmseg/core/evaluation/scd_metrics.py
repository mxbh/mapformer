import torch
from torch import Tensor
import numpy as np
from collections import OrderedDict
from .metrics import eval_metrics, total_intersect_and_union, f_score


def scd_eval_metrics(
        results,
        gt_bc_maps,
        gt_sem_maps,
        num_semantic_classes,
        ignore_index_bc,
        ignore_index_sem
    ):
    assert len(results) == len(gt_bc_maps) == len(gt_sem_maps)

    total_bc_intersect = np.zeros((1,), dtype=np.float64)
    total_bc_union = np.zeros((1,), dtype=np.float64)
    total_bc_pred = np.zeros((1,), dtype=np.float64)
    total_bc_gt = np.zeros((1,), dtype=np.float64)

    total_sc_intersect = np.zeros((num_semantic_classes, ), dtype=np.float64)
    total_sc_union = np.zeros((num_semantic_classes, ), dtype=np.float64)

    total_sem_intersect = np.zeros((num_semantic_classes, ), dtype=np.float64)
    total_sem_union = np.zeros((num_semantic_classes, ), dtype=np.float64)

    for i in range(len(results)):
        pred_bc = results[i]['bc']
        pred_sem = results[i]['sem']
        gt_bc = gt_bc_maps[i]
        gt_sem = gt_sem_maps[i]

        mask_bc = (gt_bc != ignore_index_bc)
        # mask_sem = (gt_sem != ignore_index_sem)

        pred_bc_masked = pred_bc[mask_bc]
        gt_bc_masked = gt_bc[mask_bc]
        pred_sem_masked = pred_sem[mask_bc]
        gt_sem_masked = gt_sem[mask_bc]

        # BC
        intersect_bc = np.logical_and((pred_bc_masked == 1), (gt_bc_masked == 1))
        union_bc = np.logical_or((pred_bc_masked == 1), (gt_bc_masked == 1))
        total_bc_intersect = total_bc_intersect + intersect_bc.sum()
        total_bc_union = total_bc_union + union_bc.sum()
        total_bc_pred = total_bc_pred + pred_bc_masked.sum()
        total_bc_gt = total_bc_gt + gt_bc_masked.sum()
        
        # SC
        change_mask = (gt_bc == 1)
        intersect_sc = pred_sem[change_mask][pred_sem[change_mask] == gt_sem[change_mask]]
        intersect_area_sc = np.histogram(intersect_sc, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        pred_area_sc = np.histogram(pred_sem[change_mask], bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        gt_area_sc = np.histogram(gt_sem[change_mask], bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        union_area_sc = pred_area_sc + gt_area_sc - intersect_area_sc
        total_sc_intersect = total_sc_intersect + intersect_area_sc
        total_sc_union = total_sc_union + union_area_sc

        # sem
        intersect_sem = pred_sem_masked[pred_sem_masked == gt_sem_masked]
        intersect_area_sem = np.histogram(intersect_sem, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        pred_area_sem = np.histogram(pred_sem_masked, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        gt_area_sem = np.histogram(gt_sem_masked, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        union_area_sem = pred_area_sem + gt_area_sem - intersect_area_sem
        total_sem_intersect = total_sem_intersect + intersect_area_sem
        total_sem_union = total_sem_union + union_area_sem

    ret_metrics = OrderedDict()
    ret_metrics['BC'] = (total_bc_intersect / total_bc_union).item()
    ret_metrics['SC'] = (total_sc_intersect / total_sc_union).mean()
    ret_metrics['mIoU'] = (total_sem_intersect / total_sem_union).mean()
    ret_metrics['BC_recall'] = (total_bc_intersect / total_bc_gt).item()
    ret_metrics['BC_precision'] = (total_bc_intersect / total_bc_pred).item()
    ret_metrics['SCS'] = 0.5 * (ret_metrics['BC'] + ret_metrics['SC'])
    ret_metrics['SC_per_class'] = total_sc_intersect / total_sc_union
    ret_metrics['IoU_per_class'] = total_sem_intersect / total_sem_union

    return ret_metrics
