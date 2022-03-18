# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------- 
# This file is an example of backbone. It's an simplified implementation
# of ConvNeXt modified from MMClassification.
# ---------------------------------------- 

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmcv.runner.base_module import ModuleList, Sequential

from mmcls.models import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images. """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block."""

    def __init__(self,
                 in_channels,
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.norm = LayerNorm2d(in_channels)

        mid_channels = int(mlp_ratio * in_channels)

        self.pointwise_conv1 = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Linear(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x


@BACKBONES.register_module(force=True)  # force to override class with the same name.
class ConvNeXt(BaseBackbone):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Here is an example backbone, almost copied from mmcls.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """
    # Preset architectures
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 gap_before_final_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # -------------------- Check and process arguments --------------------
        if isinstance(arch, str):
            arch = self.arch_settings[arch]

        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            out_indices[i] = self.num_stages + index if index < 0 else index
            assert 0 <= out_indices[i] < self.num_stages, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.gap_before_final_norm = gap_before_final_norm

        # -------------------- Construct modules --------------------
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            LayerNorm2d(self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = LayerNorm2d(channels)
                self.add_module(f'norm{i}', norm_layer)

    def forward(self, x):
        # Usually, the backbone is used to extract feature maps for
        # classification, detection and segmentation tasks. Therefore, we don't
        # recommand to add structures for specific tasks like global average
        # pooling and classifier head

        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x: torch.Tensor = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x).contiguous())

        # The output should be an tuple to support outputing features of
        # different stages. Even if you don't want support multi-stages
        # output, it should be a tuple with a single item.
        return tuple(outs)
