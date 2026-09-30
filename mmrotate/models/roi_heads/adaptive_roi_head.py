# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from mmdet.models.losses import SmoothL1Loss
from mmdet.models.task_modules.samplers import SamplingResult
from mmrotate.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from mmdet.models.utils import unpack_gt_instances
from mmdet.models.roi_heads import StandardRoIHead
from mmengine.logging import print_log
import statistics

EPS = 1e-15


@MODELS.register_module()
class AdaptiveRoIHead(StandardRoIHead):
    """Adaptive RoI Head"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.bbox_head.loss_bbox, SmoothL1Loss)
        # 更新间隔
        self.update_iter_interval = 10
        self.counter = 0
        
        # 更新 beta
        self.beta_topk = 10
        self.initial_beta = self.bbox_head.loss_bbox.beta
        self.beta_history = []
        
        # 更新 loss_weight
        self.init_loss_weight = self.bbox_head.loss_bbox.loss_weight
        self.min_loss_weigh =  self.bbox_head.loss_bbox.loss_weight
        self.max_loss_weigh = 5.0
        self.last_loss_k = None
        self.integral_loss_bbox = []
        self.integral_loss_cls = []

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Forward function for training.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        # TODO RECORD
        self.integral_loss_cls.append(losses['loss_cls'].detach())
        self.integral_loss_bbox.append(losses['loss_bbox'].detach())
        
        self.counter += 1
        if self.counter % self.update_iter_interval == 0:
            self.update_hyperparameters()
            self.counter = 0
        # EDN TODO: RECORD

        return losses

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])

        # TODO: RECORD
        # `bbox_targets[2]` and `bbox_targets[3]` stand for bbox_targets
        # and bbox_weights, respectively
        bbox_targets = bbox_loss_and_target['bbox_targets']
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        num_imgs = len(sampling_results)
        if num_pos > 0:
            cur_target = bbox_targets[2][pos_inds].abs().mean(dim=1)
            beta_topk = min(self.beta_topk * num_imgs, num_pos)
            cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
            self.beta_history.append(cur_target)
        # EDN TODO: RECORD
        
        return bbox_results

    def update_hyperparameters(self):
        if (not self.beta_history) or (np.median(self.beta_history) < EPS):
            new_beta = self.bbox_head.loss_bbox.beta
        else:
            new_beta = min(self.initial_beta, np.median(self.beta_history))
            print_log("before_min_beta: " + str(self.bbox_head.loss_bbox.beta))
        self.beta_history = []
        self.bbox_head.loss_bbox.beta = new_beta
        print_log("beta: "+str(self.bbox_head.loss_bbox.beta))
        
        # print_log("integral_loss_cls: " + str(self.integral_loss_cls))
        # print_log("integral_loss_bbox: " + str(self.integral_loss_bbox))
        
        # 现在比值，相对于之前的比值，变小
        new_loss_k =  self.mean_median(self.integral_loss_bbox) / self.mean_median(self.integral_loss_cls)
        
        if self.last_loss_k is not None:
            loss_weight = self.init_loss_weight * torch.sqrt(self.last_loss_k / new_loss_k)
            # 确保大于最小值，小于最大值
            tmp = self.bbox_head.loss_bbox.loss_weight + min(self.max_loss_weigh, max(self.min_loss_weigh, loss_weight))
            self.bbox_head.loss_bbox.loss_weight = tmp / 2
            print_log("before_min_loss_weight: "+str(loss_weight))
        
        self.last_loss_k = new_loss_k
        self.integral_loss_cls = []
        self.integral_loss_bbox = []
        print_log("loss_weight: " + str(self.bbox_head.loss_bbox.loss_weight))
    
    def mean(self, numbers):
        return sum(numbers) / len(numbers)
    
    def median(self, numbers):
        return statistics.median(numbers)
    
    def mean_median(self, numbers):
        return (self.mean(numbers) + self.median(numbers)) / 2