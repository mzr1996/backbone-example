from mmdet.models import BACKBONES

from ..convnext import ConvNeXt

@BACKBONES.register_module('ConvNeXt', force=True)
class ConvNeXtDet(ConvNeXt):
    # Here, we modified the default value of ConvNeXt for detection tasks. You
    # can alos do any modifications on the backbone.
    # NOTICE: we don't recommend to use this method to modifiy the backbone.
    # And it's better to use the unified implementation.
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 drop_path_rate=0.,
                 layer_scale_init_value=1.0,
                 out_indices=[0, 1, 2, 3],
                 gap_before_final_norm=False,
                 init_cfg=None):
        super().__init__(
            arch=arch,
            in_channels=in_channels,
            stem_patch_size=stem_patch_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=out_indices,
            gap_before_final_norm=gap_before_final_norm,
            init_cfg=init_cfg)
