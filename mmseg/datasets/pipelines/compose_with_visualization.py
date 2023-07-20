
import collections
import os.path as osp

from mmcv.utils import build_from_cfg
from mmseg.utils import visualize_multiple_images
from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class ComposeWithVisualization(Compose):
    """Compose multiple transforms images with saving intermedia results sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """
    
    def __init__(self, *args, if_visualize=False, save_dir=r'./tmp'):
        self.if_visualize = if_visualize
        self.save_dir = save_dir
        super().__init__(*args)

    def __call__(self, data):
        """Call function to apply transforms sequentially with saving intermedia results.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
            
            if self.if_visualize and (type(t).__name__!='LoadImagesFromFile') and (type(t).__name__!='DefaultFormatBundle') :
                print(type(t).__name__)
                if data.get("ori_filename", None) is None:
                    img_path = osp.join(self.save_dir, type(t).__name__+'_img.jpg')
                    gt_path = osp.join(self.save_dir, type(t).__name__+'_gt.jpg')
                else:
                    img_path = osp.join(self.save_dir, data['ori_filename']+'_'+type(t).__name__+'_img.jpg')
                    gt_path = osp.join(self.save_dir, data['ori_filename']+'_'+type(t).__name__+'_gt.jpg')
                visualize_multiple_images(data['img'], dst_path=img_path, channel_per_image=3)
                visualize_multiple_images(data['gt_semantic_seg'], dst_path=gt_path, channel_per_image=3)

        return data

