# dataset settings

# CICIDS2017 has 12 classes for our task
num_classes = 12
base_classes = 4
increment_classes = 2

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
