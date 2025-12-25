# optimizer
optimizer = dict(
    type='Adam',
    lr=5e-5,
    weight_decay=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='cosine',
    by_epoch=True,
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True,
)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50,
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
    num_tasks=5,
    base_classes=4,
    inc_classes=2,
    freeze_base_network=False,
    freeze_base_head=True,
    use_memory=True,
    memory_size=100,
)
