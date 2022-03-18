_base_ = [
    './_base_/default_runtime.py',    # Runtime settings
    './_base_/adamw_bs1024.py',       # Scheduler settings
    './_base_/imagenet_bs64_224.py',  # Dataset settings
]

# import the `models` package of this repository. After importing, our
# backbones are registered and can be found by the type name.
custom_imports = dict(imports='models', allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt', arch='tiny', in_channels=3),
    # Add this line if you need to do global average pooling for the outputs of
    # backbone.
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',    # Use a linear layer to do classification
        in_channels=768,         # The output channels of the backbone
        num_classes=1000,        # The number of classes of the dataset
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
