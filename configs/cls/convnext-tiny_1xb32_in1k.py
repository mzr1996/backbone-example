_base_ = './convnext-tiny_8xb128_in1k.py'

# Modify the batch size
data = dict(samples_per_gpu=32)

# Modifiy the learning rate for small batch size
optimizer = dict(lr=0.001 / 1024 * 32)
