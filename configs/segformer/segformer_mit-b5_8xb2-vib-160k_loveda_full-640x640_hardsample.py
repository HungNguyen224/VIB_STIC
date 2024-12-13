_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_240k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
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
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

dataset_type = 'LoveDADataset'
dataset_test_type = 'LoveDADataset'
data_root = 'data/loveDA'

dataset_loveDA_train = dict(
    type=dataset_type,
    data_root='data/loveDA',
    data_prefix=dict(
        img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline)
dataset_Dalat_train = dict(
    type='LoveDADataset',
    data_root='data/dalat/re-train',
    data_prefix=dict(
        img_path='images', seg_map_path='annotations'),
    pipeline=train_pipeline)
concatenate_dataset = dict(
    type='ConcatDataset',
    datasets=[dataset_loveDA_train, dataset_Dalat_train])
train_dataloader = dict(
    batch_size=TRAIN_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dataset_loveDA_train
)
val_dataloader = dict(
    batch_size=VAL_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val',
                         seg_map_path='ann_dir/val'),
        pipeline=val_pipeline))


test_dataloader = dict(
    batch_size=VAL_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png')))
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
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000),
        in_channels=[64, 128, 320, 512],
        num_classes=NUM_CLASSES,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
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

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=10000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU'),
    logger=[dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
            dict(type='MlflowLoggerHook', interval=50, 
                                          by_epoch=False)],)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
randomness = dict(seed=304)
default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),)
