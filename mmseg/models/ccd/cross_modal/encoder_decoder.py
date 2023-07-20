from mmcv.runner import auto_fp16
from ..encoder_decoder import EncoderDecoderCCD
from ...cd.fhd import split_images
from ...builder import SEGMENTORS

@SEGMENTORS.register_module()
class EncoderDecoderCMCD(EncoderDecoderCCD):
    '''
    Overall model class for Cross-modal CD.
    '''
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, gt_semantic_seg_pre=None, gt_semantic_seg_post=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        
        if return_loss:
            img1, img2 = split_images(img)
            return self.forward_train(
                img=img2,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg_post=gt_semantic_seg_post, 
                **kwargs
                )
        else:
            assert isinstance(img, list) and len(img) == 1, 'Expected a one item list!'
            img1, img2 = split_images(img[0])
            return self.forward_test(
                imgs=[img2],
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                **kwargs
            )