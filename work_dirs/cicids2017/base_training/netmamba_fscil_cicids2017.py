/root/autodl-tmp/NewModel/configs/_base_/models/netmamba_etf.py
# model settings
model = dict(
    type='NetMambaFSCIL',
    backbone=dict(
        type='NetMamba',
        embed_dim=256,
        num_layers=4,
        ssm_cfg=dict(
            expand=2,
            d_state=16,
            dt_rank='auto',
        ),
    ),
    neck=dict(
        type='DualSelectiveSSMProjector',
        in_channels=256,
        out_channels=512,
        mid_channels=512,
        d_state=256,
        d_rank=64,
        ssm_expand_ratio=2,
        num_layers=2,
        num_layers_new=2,
        feat_size=1,
        use_new_branch=True,
        loss_weight_supp=0.001,
        loss_weight_supp_novel=0.001,
        loss_weight_sep=0.001,
        loss_weight_sep_new=0.001,
        param_avg_dim='0-1',
        detach_residual=False,
    ),
    head=dict(
        type='ETFHead',
        num_classes=15,
        base_classes=5,
        in_channels=512,
        with_bn=False,
        with_avg_pool=False,
    )
)

/root/autodl-tmp/NewModel/configs/_base_/datasets/cicids2017_fscil.py
# dataset settings

# CICIDS2017 has 15 classes
num_classes = 15
base_classes = 5
increment_classes = 10

# Data pipelines
train_pipeline = [
    dict(type='LoadTrafficData'),
    dict(type='NormalizeTraffic', mean=0.0, std=1.0),
    dict(type='ToTensor', keys=['data', 'label']),
    dict(type='Collect', keys=['data', 'label']),
]

val_pipeline = [
    dict(type='LoadTrafficData'),
    dict(type='NormalizeTraffic', mean=0.0, std=1.0),
    dict(type='ToTensor', keys=['data', 'label']),
    dict(type='Collect', keys=['data', 'label']),
]

test_pipeline = [
    dict(type='LoadTrafficData'),
    dict(type='NormalizeTraffic', mean=0.0, std=1.0),
    dict(type='ToTensor', keys=['data', 'label']),
    dict(type='Collect', keys=['data', 'label']),
]

# Dataset configurations
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='CICIDS2017Dataset',
        data_prefix='/root/autodl-tmp/NewModel/data/CICIDS2017',
        pipeline=train_pipeline,
        num_classes=num_classes,
        base_classes=base_classes,
        split='train',
    ),
    val=dict(
        type='CICIDS2017Dataset',
        data_prefix='/root/autodl-tmp/NewModel/data/CICIDS2017',
        pipeline=val_pipeline,
        num_classes=num_classes,
        base_classes=base_classes,
        split='valid',
    ),
    test=dict(
        type='CICIDS2017Dataset',
        data_prefix='/root/autodl-tmp/NewModel/data/CICIDS2017',
        pipeline=test_pipeline,
        num_classes=num_classes,
        base_classes=base_classes,
        split='test',
    ),
    train_dataloader=dict(persistent_workers=True),
    val_dataloader=dict(persistent_workers=True),
    test_dataloader=dict(persistent_workers=True),
)

# Evaluation settings
evaluation = dict(
    interval=1,
    metric='accuracy',
    metric_options=dict(topk=(1, 5)),
    save_best='accuracy_top-1',
    classwise=True,
)

/root/autodl-tmp/NewModel/configs/_base_/schedules/cicids2017_100e.py
# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    step=[30, 60, 80],
    gamma=0.1,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True,
)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=100,
)

# checkpoint settings
checkpoint_config = dict(
    interval=10,
    save_last=True,
    save_best='accuracy_top-1',
    max_keep_ckpts=5,
)

# log settings
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ],
)

# distributed training settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# incremental learning settings
incremental_learning = dict(
    start_epoch=0,
    inc_epochs=20,
    num_tasks=3,
    base_classes=5,
    inc_classes=5,
    freeze_base_network=False,
    freeze_base_head=True,
    use_memory=True,
    memory_size=100,
)

/root/autodl-tmp/NewModel/configs/cicids2017/netmamba_fscil_cicids2017.py
# NetMamba-FSCIL configuration for CICIDS2017
_base_ = [
    '../_base_/models/netmamba_etf.py',
    '../_base_/datasets/cicids2017_fscil.py',
    '../_base_/schedules/cicids2017_100e.py',
]

# model settings
model = dict(
    neck=dict(
        # Update neck parameters for CICIDS2017
        in_channels=256,
        out_channels=512,
    ),
    head=dict(
        # Update head parameters for CICIDS2017
        num_classes=15,
        base_classes=5,
    )
)

# data settings
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
)

# training settings
train_cfg = dict(
    type='IncrementalTrainLoop',
    base_epochs=80,
    inc_epochs=20,
    num_tasks=3,
    base_classes=5,
    inc_classes=5,
    eval_interval=1,
)

# evaluation settings
evaluation = dict(
    interval=1,
    metric='accuracy',
    metric_options=dict(topk=(1,)),
    save_best='accuracy_top-1',
    classwise=True,
)

# runtime settings
log_level = 'INFO'
checkpoint_config = dict(interval=10, max_keep_ckpts=5)
work_dir = './work_dirs/cicids2017/netmamba_fscil_cicids2017'
load_from = None
resume_from = None

# wandb settings
# wandb_config = dict(
#     project='NetMamba-FSCIL',
#     name='netmamba_fscil_cicids2017',
#     tags=['cicids2017', 'netmamba', 'fscil'],
# )
