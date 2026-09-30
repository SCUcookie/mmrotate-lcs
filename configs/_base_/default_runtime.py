default_scope = 'mmrotate'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=1),
    checkpoint=dict(interval=12, max_keep_ckpts=3, type='CheckpointHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend')  # https://mmdetection.readthedocs.io/zh_CN/dev-3.x/user_guides/visualization.html
]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# EMA
# custom_hooks = [
#     dict(type='mmdet.NumClassCheckHook'),
#     dict(
#         type='EMAHook',
#         ema_type='mmdet.ExpMomentumEMA',
#         momentum=0.0002,
#         update_buffers=True,
#         priority=49)
# ]