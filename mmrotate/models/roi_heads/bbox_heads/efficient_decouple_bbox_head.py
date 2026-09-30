# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmdet.models.backbones.resnet import Bottleneck
from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType, OptMultiConfig
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmrotate.models.backbones.lsknet import LSKblock, Mlp, DWConv
from mmdet.models.layers.inverted_residual import InvertedResidual
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch.nn.init import xavier_uniform_, constant_
from torch.nn import Linear, Dropout, ReLU
import torch
from torch.nn import MultiheadAttention

init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform', 
                 ),
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='Kaiming', layer='Linear', a=0, mode='fan_out', nonlinearity='relu'),
        ]

class AttentionBlock(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim, dilation=1)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class Space(BaseModule):
    def __init__(self,
                 dim: int,
                 init_cfg: OptMultiConfig = init_cfg) -> None:
        super().__init__(init_cfg=init_cfg)
        self.attention = AttentionBlock(dim)
        # identity path
        self.conv_identity = ConvModule(
            dim,
            dim,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        
        identity = x
        x = self.attention(x)
        identity = self.conv_identity(identity)
        out = x + identity
        
        return out

class Channel(BaseModule):
    def __init__(self,
                dim: int,
                ratio = 1,
                init_cfg: OptMultiConfig = init_cfg) -> None:
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        x = self.conv(x + x*out)
        return x

class Task(BaseModule):
    def __init__(self,
                dim: int,
                out_dim: int,
                num_heads = 8,
                init_cfg: OptMultiConfig = init_cfg) -> None:
        super().__init__(init_cfg=init_cfg)
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        if out_dim == dim:
            self.conv_dim = nn.Identity()
        else:
            self.conv_dim = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x):
        copy = x
        x = x.mean(dim=(2, 3))  # (N, C)
        x = x.unsqueeze(1)  # (N, 1, C)
        x, _ = self.mha(x, x, x)  # (N, 1, C)
        x = self.conv_dim(x)
        x = x.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        x = x * copy
        return x

@MODELS.register_module()
class EfficientDecoupleConvFCBBoxHead(BBoxHead):
    def __init__(self,
                 num_convs: int = 2,
                 num_fcs: int = 2,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='SyncBN'),
                 init_cfg: MultiConfig = [
                    # dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                    # dict(type='Xavier', layer='Conv1d', distribution='uniform'),
                    # dict(type='Kaiming', layer='Linear', a=0, mode='fan_out', nonlinearity='relu'),
                    dict(
                        type='Normal',
                        override=[
                            dict(type='Normal', name='fc_cls', std=0.01),
                            dict(type='Normal', name='fc_reg', std=0.001),
                            dict(type='Xavier', name='fc_branch', distribution='uniform')
                        ]
                    )
                ],
                 **kwargs) -> None:
        kwargs.setdefault('with_avg_pool', True)
        super().__init__(init_cfg=init_cfg, **kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        # self.res_block = LSKResBlock(self.in_channels, self.conv_out_channels)
        self.res_block = nn.Sequential(
            Space(self.in_channels),
            Channel(self.in_channels),
            Task(dim=self.in_channels, out_dim=self.in_channels),
        )

        # add conv heads
        self.conv_branch = self._add_conv_branch()

        # add fc heads
        self.fc_branch = self._add_fc_branch()

        if 'predict_box_type' in kwargs and kwargs['predict_box_type'] == 'rbox':
            out_dim_reg = 5 if self.reg_class_agnostic else 5 * self.num_classes
        else: 
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU()

    def _add_conv_branch(self) -> None:
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                InvertedResidual(
                    in_channels=self.conv_out_channels,
                    out_channels=self.conv_out_channels,
                    mid_channels=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self) -> None:
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def forward(self, x_cls: Tensor, x_reg: Tensor) -> Tuple[Tensor]:
        # conv head
        x_conv = self.res_block(x_reg)
        for conv in self.conv_branch:
            x_conv = conv(x_conv)
        x_conv = self.avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        x_angle = x_fc
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))
        cls_score = self.fc_cls(x_fc)
        
        return cls_score, bbox_pred