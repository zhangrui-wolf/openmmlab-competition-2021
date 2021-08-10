_base_ = './repvggA0_b64x4_imagenet.py'

model = dict(backbone=dict(arch='A2'), head=dict(in_channels=1408))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25)
runner = dict(max_epochs=120)
