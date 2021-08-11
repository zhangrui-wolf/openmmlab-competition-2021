from mmcls.models.utils.se_layer import SELayer
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class SEPlugin(SELayer):

    def __init__(self, **kwargs):
        super(SEPlugin, self).__init__(**kwargs)
