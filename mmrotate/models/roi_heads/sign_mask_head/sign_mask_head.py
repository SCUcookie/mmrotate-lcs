# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from typing import List, Tuple
from .decode_head import BaseDecodeHead
from mmrotate.registry import MODELS
from .wrappers import resize
from torch import Tensor
import cv2
import numpy as np
from mmrotate.structures.bbox import rbox2qbox

@MODELS.register_module()
class SignMaskHead(BaseDecodeHead):
    """The all mlp Head of SignMaskHead.

    This head is the implementation of SignMaskHead
    `Drawing on Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
    def forward(self, inputs):
        # Receive 5 stage backbone feature map: 1/4, 1/8, 1/16, 1/32 1/64
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
    
    def _stack_batch_gt(self, batch_data_samples) -> Tensor:
        gt_semantic_segs = [
            data_sample.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples,
             train_cfg) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        
        batch_data_masks = self.bbox_to_mask(batch_data_samples)

        
        losses = self.loss_by_feat(seg_logits, batch_data_masks)
        return losses
    
    def draw_polygon_on_mask(self, mask, corners):
        """
        使用 PyTorch 在 mask 上绘制多个多边形。
        参数:
            mask: 目标 mask，形状为 (H, W)。
            corners: 旋转矩形的角点，形状为 (n, 8)，每行表示一个旋转矩形的四个角点。
        """
        device = mask.device # 获取 mask 的设备
        if mask.device.type == "cuda":
            corners = corners.cpu().numpy()
            corners = corners.reshape(-1, 4, 2).astype(np.int32)
            mask_np = mask.cpu().numpy()
            # 使用 OpenCV 在 mask 上填充每个多边形
            for i, cor in enumerate(corners):
                cv2.fillPoly(mask_np, [cor], color=1)  # 填充多边形
            return torch.from_numpy(mask_np).to(mask.device)
        else:
            corners = corners.numpy()
            mask_np = mask.numpy()
            for i, cor in enumerate(corners):
                cv2.fillPoly(mask_np, [cor], color=1)  # 填充多边形
            return torch.from_numpy(mask_np)

    def bbox_to_mask(self, batch_data_samples):
        """
        从 batch_data_samples 中提取旋转框并生成 mask。
        """
        pad_shape = batch_data_samples[0].get("pad_shape")
        mask_shape = [s // 4 for s in pad_shape]  # 适当缩小尺寸用于 mask
        B = len(batch_data_samples)
        H_img, W_img = pad_shape
        H_mask, W_mask = mask_shape

        device = batch_data_samples[0].get("gt_instances").bboxes.tensor.device
        scale_x = W_mask / W_img
        scale_y = H_mask / H_img

        # 初始化 mask
        masks = torch.zeros((B, H_mask, W_mask), dtype=torch.float32, device=device)

        for i, data_sample in enumerate(batch_data_samples):
            gt_instances = data_sample.get("gt_instances")
            bboxes_tensor = gt_instances.bboxes.tensor
            qbox = rbox2qbox(bboxes_tensor)
            qbox = qbox * scale_y
            
            masks[i] = self.draw_polygon_on_mask(masks[i], qbox)
            
        masks = masks.long()
        
        # for i, m in enumerate(masks):
        #     # 保存 mask 为 PNG 图片
        #     from PIL import Image
        #     from datetime import datetime
        #     mask_image = Image.fromarray((m.cpu().numpy() * 255).astype('uint8'))
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = f"out_dir/final_mask_{timestamp}.png"
        #     mask_image.save(filename)
            
        return masks[:, None, ...]