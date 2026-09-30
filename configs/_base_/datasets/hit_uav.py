# dataset settings
dataset_type = 'HITUAVDataset'
data_root = '../../data-5/'
backend_args = None

# 需要 Pad
__K__ = 0.5
__scale__ = (int(640/__K__), int(512/__K__))
__test_scale__ = __scale__
batch_size = 16


train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=__scale__, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180),
    dict(type='mmdet.Pad', size=__scale__, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=__scale__, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Pad', size=__scale__, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=__test_scale__, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Pad', size=__test_scale__, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=batch_size,  # 4 24G
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=True, # 锁页内存 提供 data copy 速度
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/trainval.txt',
        ann_subdir='Annotations/',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=2*batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/test.txt',
        ann_subdir='Annotations/',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = dict(  # test_dataloader == val_dataloader 只是分辨率设置不同
    batch_size=2*batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/test.txt',
        ann_subdir='Annotations/',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = [
    dict(
        type='DOTAMetric',
        eval_mode='11points',
        prefix='dota_ap07',
        metric='mAP'),
    dict(
        type='DOTAMetric',
        eval_mode='area',
        prefix='dota_ap12',
        metric='mAP'),
]
# val_evaluator = dict(type='DOTAMetric', metric='mAP') // 默认的 07
test_evaluator = val_evaluator
