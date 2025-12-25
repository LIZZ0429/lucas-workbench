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
        num_classes=12,
        base_classes=4,
    )
)

# data settings
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
)

# training settings
train_cfg = dict(
    type='IncrementalTrainLoop',
    base_epochs=80,
    inc_epochs=20,
    num_tasks=5,
    base_classes=4,
    inc_classes=2,
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
