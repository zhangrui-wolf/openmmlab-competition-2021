_base_ = './repvggA0_b64x4_imagenet.py'

model = dict(backbone=dict(arch='B0'), head=dict(in_channels=1280))
