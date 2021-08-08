_base_ = '../repvggB2g4_64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
