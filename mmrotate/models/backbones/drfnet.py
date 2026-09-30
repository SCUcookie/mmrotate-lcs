import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import (BaseModule, constant_init, normal_init,
                            trunc_normal_init)
from torch.nn.modules.utils import _pair as to_2tuple
from mmrotate.registry import MODELS
try:
    from mamba_ssm import Mamba
except ImportError:
    print('Mamba not found, please install mamba_ssm first')
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba

class MambaBlock(BaseModule):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.attn = MambaLayer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(x))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim, use_mamba=False):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.use_mamba = use_mamba
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

        if self.use_mamba:
            self.mamba = MambaBlock(dim)
            self.gate_layer = nn.Sequential(
                nn.Linear(3*dim, 3, bias=False),
                nn.Softmax(dim=-1)
            )      
            groups_K = 8
            self.convx1 = nn.Conv2d(dim, dim, 1, groups=dim//groups_K)
            self.convx2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim//groups_K)
            self.convx3 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim//groups_K)
            self.norm1 = nn.SyncBatchNorm(dim)
            self.norm2 = nn.SyncBatchNorm(dim)
            self.norm3 = nn.SyncBatchNorm(dim)
            self.weights = nn.Parameter(torch.ones(6))
                  
    def forward(self, x):
        if self.use_mamba:
            # x = self.mamba(x) * self.weights[0] + x * self.weights[1]
            x = self.forward_mamba(x) * self.weights[0] + x * self.weights[1] 

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
        x = x * attn

        if self.use_mamba:
            x = (
                    self.weights[2] * self.norm1(self.convx1(x)) +
                    self.weights[3] * self.norm2(self.convx2(x)) +
                    self.weights[4] * self.norm3(self.convx3(x)) +
                    self.weights[5] * x
                )
            
        return x

    def forward_mamba(self, x):
        B = x.shape[0]
        x_inputs = [x, torch.flip(x, [1])]
        rand_index = torch.randperm(x.size(1))
        x_inputs.append(x[:, rand_index])
        x_inputs = torch.cat(x_inputs, dim=0)
        # x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
        x_inputs = self.mamba(x_inputs)
        forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
        reverse_x = torch.flip(reverse_x, [1])
        # reverse the random index
        rand_index = torch.argsort(rand_index)
        shuffle_x = shuffle_x[:, rand_index]
        mean_forward_x = torch.mean(forward_x, dim=(2, 3))
        mean_reverse_x = torch.mean(reverse_x, dim=(2, 3))
        mean_shuffle_x = torch.mean(shuffle_x, dim=(2, 3))
        gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
        gate = self.gate_layer(gate)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x
        return x


class Attention(nn.Module):
    def __init__(self, d_model, use_mamba=False):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model, use_mamba)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None, use_mamba=False):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, use_mamba)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W

@MODELS.register_module()
class DRFNet(BaseModule):
    """Dynamic receptive field feature extraction network.
    Args:
        in_chans (int): The num of input channels. Defaults to 3.
        embed_dims (List[int]): Embedding channels of each LSK block.
            Defaults to [64, 128, 256, 512]
        mlp_ratios (List[int]): Mlp ratios. Defaults to [8, 8, 4, 4]
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        depths (List[int]): Number of LSK block in each stage.
            Defaults to [3, 4, 6, 3]
        num_stages (int): Number of stages. Defaults to 4
        pretrained (bool): If the model weight is pretrained. Defaults to None,
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to None.
    """

    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super(DRFNet, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.lsk_index = [0, 1, 2, 3]
        self.use_mambas = [False, False, True, True]
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg)

            if i in self.lsk_index:
                block = nn.ModuleList([
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        norm_cfg=norm_cfg,
                        use_mamba=self.use_mambas[i]
                        ) for j in range(depths[i])
                ])
            else:
                block = nn.ModuleList([
                    MambaBlock(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        norm_layer=nn.LayerNorm,  # 默认
                        sr_ratio=1,  # 没有用
                        ) for j in range(depths[i])
                ])

            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(DRFNet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x