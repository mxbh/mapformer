import numpy as np
import os.path as osp

from joblib import delayed, Parallel

import mmcv
import torch


def fast_single_gpu_test_sp(model,
                            data_loader,
                            out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        batch_size = len(result)
        img_metas = data['img_metas'][0].data[0]
        for index, img_meta in enumerate(img_metas):
            if out_dir:
                h, w, _ = img_meta['img_shape']
                out_file = osp.join(out_dir, img_meta['ori_filename'])

                seg = result[index]
                palette = np.array(dataset.PALETTE)
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color
                # convert to BGR
                color_seg = color_seg[..., ::-1]
                img = color_seg
                img = img.astype(np.uint8)

                if out_file is not None:
                    mmcv.imwrite(img, out_file)

        for _ in range(batch_size):
            prog_bar.update()


def fast_single_gpu_test_mp(model,
                            data_loader,
                            out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        batch_size = len(result)
        if out_dir:
            img_metas = data['img_metas'][0].data[0]
            Parallel(n_jobs=batch_size)(
                delayed(_test_module)(
                    img_meta, out_dir, result, index, dataset
                )
                for index, img_meta in enumerate(img_metas)
            )

        for _ in range(batch_size):
            prog_bar.update()


def _test_module(img_meta, out_dir, result, index, dataset):
    h, w, _ = img_meta['img_shape']
    out_file = osp.join(out_dir, img_meta['ori_filename'])

    seg = result[index]
    palette = np.array(dataset.PALETTE)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    img = color_seg
    img = img.astype(np.uint8)
    mmcv.imwrite(img, out_file)
