_base_ = './convnext-tiny_8xb128_in1k.py'

# Modification on the models settings defined in the base setting
model = dict(backbone=dict(arch='base'), head=dict(in_channels=1024))
