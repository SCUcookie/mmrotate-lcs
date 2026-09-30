import argparse
import torch
import torch.nn as nn
from torchstat import stat

# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmengine.config import Config, DictAction

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_tensor):
    flops = 0

    def flops_hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            batch_size, out_channels, out_height, out_width = output.size()
            kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            flops += kernel_size * out_channels * out_height * out_width / module.groups
        elif isinstance(module, nn.Linear):
            flops += input[0].numel() * output.size(1)

    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(flops_hook))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return flops

def main():
    parser = argparse.ArgumentParser(description='Count model parameters and FLOPs')
    parser.add_argument('--config', default='configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_drfnet.py', help='train config file path')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[3, 1024, 1024], help='Input tensor shape (C, H, W)')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    # 输入张量
    input_tensor = torch.randn(1, *args.input_shape).cuda()  # Move input tensor to GPU

    # 计算参数量和计算量
    num_params = count_parameters(model)
    num_flops = count_flops(model, input_tensor)

    print(f"模型参数总数: {num_params}")
    print(f"模型计算量 (FLOPs): {num_flops}")

if __name__ == '__main__':
    main()
