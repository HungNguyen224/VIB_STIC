_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    # '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_240k.py']

checkpoint = 'models/segformer/mit_b5_20220624-658746d9.pth'
NUM_CLASSES = 7
TRAIN_BATCH = 4
VAL_BATCH = 2

# dataset settings
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

dataset_type = 'LoveDADataset'
dataset_test_type = 'LoveDADataset'
data_root = 'data/loveDA/'

dataset_loveDA_train = dict(
    type=dataset_type,
    data_root='data/loveDA',
    data_prefix=dict(
        img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline)
dataset_Dalat_train = dict(
    type='LoveDADataset',
    data_prefix=dict(
        img_path='data/dalat/re-train/images/', seg_map_path='data/dalat/re-train/annotations/'),
    pipeline=train_pipeline)
concatenate_dataset = dict(
    type='ConcatDataset',
    datasets=[dataset_loveDA_train, dataset_Dalat_train])
train_dataloader = dict(
    batch_size=TRAIN_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=concatenate_dataset
)

val_dataloader = dict(
    batch_size=VAL_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root='data/loveDA',
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=VAL_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="LoveDADataset",
        data_prefix=dict(img_path='images', seg_map_path='annotations'),
        data_root='data/dalat/test',
    pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoderVIB',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='SegformerVIBHead',
        in_channels=[64, 128, 320, 512],
        num_classes=NUM_CLASSES,
        loss_decode=dict(
            type='FocalLoss',
            gamma=1.9,
            alpha=0.7,
            use_sigmoid=True,
            loss_weight=1.0),
        loss_vib=dict(
            type='KLLoss',
            reduction_override='mean',
            loss_weight=0.05)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=10000)
default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU'))
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
randomness = dict(seed=304)


