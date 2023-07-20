from typing import OrderedDict
import torch
from torch import nn
from copy import deepcopy

class EMAWrapper(nn.Module):
    '''
    Wrapper around any Segmentor architecture that allows to keep track of EMA weights and do evaluation with the EMA model.
    '''
    def __init__(self, model, keep_rate=0.9996, ema_eval=True):
        super(EMAWrapper, self).__init__()
        self.model = model
        self.ema_model = deepcopy(model)
        assert 0 <= keep_rate <= 1
        self.keep_rate = keep_rate
        self.ema_eval = ema_eval

    @property
    def CLASSES(self):
        if hasattr(self.model, 'CLASSES'):
            return self.model.CLASSES
        else: 
            return None

    @CLASSES.setter
    def CLASSES(self, classes):
        self.model.CLASSES = classes
        self.ema_model.CLASSES = classes
    

    @property
    def PALETTE(self):
        if hasattr(self.model, 'PALETTE'):
            return self.model.PALETTE
        else: 
            return None

    @PALETTE.setter
    def PALETTE(self, palette):
        self.model.PALETTE = palette
        self.ema_model.PALETTE = palette

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self.model, 'neck') and self.model.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self.model,
                       'auxiliary_head') and self.model.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self.model, 'decode_head') and self.model.decode_head is not None

    def extract_feat(self, *args, **kwargs):
        if self.ema_eval and not self.training:
            return self.ema_model.extract_feat(*args, **kwargs)
        else:
            return self.model.extract_feat(*args, **kwargs)


    def encode_decode(self, *args, **kwargs):
        if self.ema_eval and not self.training:
            return self.ema_model.encode_decode(*args, **kwargs)
        else:
            return self.model.encode_decode(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def simple_test(self, *args, **kwargs):
        if self.ema_eval:
            return self.ema_model.simple_test(*args, **kwargs)
        else: 
            return self.model.simple_test(*args, **kwargs)

    def aug_test(self, *args, **kwargs):
        if self.ema_eval:
            return self.ema_model.aug_test(*args, **kwargs)
        else: 
            return self.model.aug_test(*args, **kwargs)

    def init_weights(self, *args, **kwargs):
        self.model.init_weights(*args, **kwargs)
        self.ema_model.load_state_dict(self.model.state_dict())

    def forward_test(self, *args, **kwargs):
        if self.ema_eval:
            return self.ema_model.forward_test(*args, **kwargs)
        else: 
            return self.model.forward_test(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.ema_eval and not self.training:
            return self.ema_model.forward(*args, **kwargs)
        else:
            return self.model.forward(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        outputs = self.model.train_step(*args, **kwargs)
        # doing the update here (before the model update) causes the EMA to lag on step behind
        # --> ignore because one step with a keep rate > 0.999 is negligible
        self.update_ema_model() 
        return outputs

    def val_step(self, *args, **kwargs):
        if self.ema_eval:
            return self.ema_model.val_step(*args, **kwargs)
        else: 
            return self.model.val_step(*args, **kwargs)

    def show_result(self, *args, **kwargs):
        return self.model.show_result(*args, **kwargs)

    @torch.no_grad()
    def update_ema_model(self):
        '''
        Taken from https://github.com/facebookresearch/unbiased-teacher/blob/226f8c4b7b9c714e52f9bf640da7c75343803c52/ubteacher/engine/trainer.py#L632
        '''
        main_model_dict = self.model.state_dict()
        new_ema_dict = OrderedDict()
        for key, value in self.ema_model.state_dict().items():
            if key in main_model_dict.keys():
                new_ema_dict[key] = (main_model_dict[key] * (1 - self.keep_rate) + value * self.keep_rate)
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_model.load_state_dict(new_ema_dict)