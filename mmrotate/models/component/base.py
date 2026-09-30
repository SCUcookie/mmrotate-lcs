import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
# from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
#                                         trunc_normal_init)
from mmengine.model import constant_init, normal_init,trunc_normal_init
from mmrotate.registry import MODELS
# from mmrotate.models.builder import BACKBONES
from mmengine.model import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class DWConv(nn.Module):
    def __init__(self, ch):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(ch, ch, 3, 1, 1, bias=True, groups=ch)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class DWSConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(DWSConv, self).__init__()
        self.dwconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=1, groups=ch_in, bias=True)
        self.pconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pconv(x)
        x = self.act(x)
        return x
