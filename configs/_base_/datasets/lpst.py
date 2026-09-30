# dataset settings
dataset_type = 'LPSTDataset'
data_root = '../../data-6/'    # 删除空数据 （mmrotate会自己排除， 因此可以不删除） val map 高
backend_args = None

__K__ = 2
__scale__ = (640/__K__, 512/__K__)
__test_scale__ = (640/__K__, 512/__K__)
batch_size = 32

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=__scale__, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    # dict(
    #         type='RandomRotate',
    #         prob=0.5,
    #         angle_range=180,
    #         rect_obj_labels=[9, 11]),
    dict(type='mmdet.PackDetInputs',
        # meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
        #                     'scale_factor',)
         )
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=__scale__, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval_annfiles/',
        data_prefix=dict(img_path='images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2*batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='trainval/val_annfiles/',
        ann_file='val_annfiles/',
        data_prefix=dict(img_path='images/'),
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = [
    dict(
        type='DOTAMetric',
        iou_thrs=0.5,
        scale_ranges=None,
        metric='mAP',
        eval_mode='11points',
        prefix='dota_ap07'),
    dict(
        type='DOTAMetric',
        iou_thrs=0.5,
        scale_ranges=None,
        metric='mAP',
        eval_mode='area',
        prefix='dota_ap12'),
]



# test_dataloader = val_dataloader
# test_evaluator = val_evaluator


# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=__test_scale__, keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_dataloader = dict(
    batch_size=2*batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix=dict(img_path='test_less/images/'),
        data_prefix=dict(img_path='images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='DOTAMetric',
    predict_box_type='rbox',
    format_only=True,
    outfile_prefix='./work_dirs/dota/Task1',
    merge_patches=True, # 合并 patchs
    iou_thr=0.1,
)
