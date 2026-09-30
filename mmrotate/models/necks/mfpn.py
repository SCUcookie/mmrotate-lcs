import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS
from mmpretrain.models.utils import build_norm_layer, to_2tuple
from mmdet.models.necks import FPN
from typing import List, Tuple, Union
from torch import Tensor
from .component.mamba import MLayer
from .component.compute import IntensiveCompute, \
    ChannelAttention, SpatialAttention, SelectiveFeatureFusion, \
    FeatureSelectionModule, FeatureInteractionModule
from .component.dynamic_filter import DynamicFilter
from .component.carafe import CARAFE

@MODELS.register_module()
class MFPN(FPN):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform', 
                 ),
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='Kaiming', layer='Linear', a=0, mode='fan_out', nonlinearity='relu'),
            dict(type='Constant', val=1, layer='BatchNorm2d'),
        ]
    ):
        super(MFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg
        )

        # self.mlayers = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level-2):
        #     mlayer = MLayer(
        #         num_layers=1,
        #         in_channels=256,
        #         img_size=img_size//(16 * 2**i),
        #         embed_dims=256,
        #         # patch_size=8//(2**i) if (8//(2**i))>=1 else 1,
        #         patch_size=1,
        #         pe_type='learnable',
        #         # pe_type='sine',
        #         drop_rate=0.,
        #         norm_cfg=dict(type='LN', eps=1e-6),
        #         patch_cfg=dict(),
        #         layer_cfgs=dict(),
        #         init_cfg=init_cfg)
        #     self.mlayers.append(mlayer)

        # self.intensive_computes = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level-1):
        #     intensive_compute = IntensiveCompute(out_channels)
        #     self.intensive_computes.append(intensive_compute)

        # self.dynamic_filters = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level):
        #     dynamic_filter = DynamicFilter(
        #         input_dim=out_channels,
        #         filter_size=img_size//(4 * 2**i),
        #         num_filters=4,
        #         init_cfg=init_cfg
        #     )
        #     self.dynamic_filters.append(dynamic_filter)

        # self.carafes = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level-1):
        #     carafe = CARAFE(
        #         in_channels=out_channels,
        #         out_channels=out_channels,
        #         kernel_size=3,
        #         up_kernel_size=3,
        #         scale_factor=2,
        #     )
        #     self.carafes.append(carafe)
        # self.carafes = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level-1):
        #     # 构建一个顺序的亚像素卷积模块
        #     subpixel_conv = nn.Sequential(
        #         nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1),
        #         nn.PixelShuffle(2)
        #     )
        #     self.carafes.append(subpixel_conv)
        
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fsm = FeatureSelectionModule(in_channels[i], out_channels)
            self.lateral_convs.append(fsm)
            
        self.fims = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            fim = FeatureInteractionModule(out_channels)
            self.fims.append(fim)

        # self.sff_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3,
        #                           padding=1, stride=2)
        # self.sffs = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level-1):
        #    sff =  SelectiveFeatureFusion(out_channels, scale_factor = 2**i)
        #    self.sffs.append(sff)
        # 删除 fpn_convs 属性
        # del self.fpn_convs
           

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        if hasattr(self, 'sffs'):
            laterals = [
                sff_fsm(inputs[i + self.start_level])
                for i, sff_fsm in enumerate(self.sff_fsms)
            ]
            used_backbone_levels = len(laterals)

            for i in range(0, used_backbone_levels-1):
                laterals[i] = self.sffs[used_backbone_levels-2-i](laterals[-1], laterals[i])

            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
            outs.append(self.sff_conv(outs[-1]))
            return tuple(outs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)   
        
        # intensive_computes on down-top path
        if hasattr(self, 'intensive_computes'):
            for i in range(0, used_backbone_levels-1):
                laterals[i+1] = self.intensive_computes[i](laterals[i]) + laterals[i+1]


        # dynamic_filters 
        if hasattr(self, 'dynamic_filters'):
            laterals[0] = self.dynamic_filters[0](laterals[0])
            for i in range(0, used_backbone_levels-1):
                tmp = F.interpolate(laterals[i], size=laterals[i+1].shape[2:], mode='bilinear')
                laterals[i+1] = self.dynamic_filters[i+1](laterals[i+1]) + tmp
            # for i in range(0, used_backbone_levels):
            #     laterals[i] = self.dynamic_filters[i](laterals[i])


        # mamba layers
        if hasattr(self, 'mlayers'):
            for i in range(0, used_backbone_levels-2):
                laterals[i+2] = self.mlayers[i](laterals[i+2])
                # laterals[i] = self.mlayers[i](laterals[i])
            

        # build top-down path
        if hasattr(self, 'carafes'):
            for i in range(used_backbone_levels-1, 0, -1):
                laterals[i-1] = laterals[i-1] + self.carafes[i-1](laterals[i])
        else:
            for i in range(used_backbone_levels - 1, 0, -1):
                # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
                #  it cannot co-exist with `size` in `F.interpolate`.
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)


        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        if hasattr(self, 'fims'):
            for i in range(0, used_backbone_levels-1):
                outs[i+1] = self.fims[i](outs[i], outs[i+1])
        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)