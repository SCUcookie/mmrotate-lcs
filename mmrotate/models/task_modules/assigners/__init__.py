# Copyright (c) OpenMMLab. All rights reserved.
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .rotate_iou2d_calculator import (FakeRBboxOverlaps2D,
                                      QBbox2HBboxOverlaps2D,
                                      RBbox2HBboxOverlaps2D, RBboxOverlaps2D)
from .rotated_atss_assigner import RotatedATSSAssigner
from .sas_assigner import SASAssigner
from .hierarchical_assigner import HieAssigner
from .metric_calculator import RBbox2HBboxDistanceMetric
from .max_sim_assigner import MaxSiMAssigner
from .sim2d_calculator import RBbox2HBboxSiM2D
from .ratss_assigner import RATSSAssigner
from .rdynamic_soft_label_assigner import RDynamicSoftLabelAssigner
from .rfla_mode_switch_hook import RFLAModeSwitchHook

__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner',
    'RotatedATSSAssigner', 'RBboxOverlaps2D', 'FakeRBboxOverlaps2D',
    'RBbox2HBboxOverlaps2D', 'QBbox2HBboxOverlaps2D', 'HieAssigner', 'RBbox2HBboxDistanceMetric', 'RBbox2HBboxSiM2D',
    'RATSSAssigner', 'RDynamicSoftLabelAssigner', 'RFLAModeSwitchHook'
]
