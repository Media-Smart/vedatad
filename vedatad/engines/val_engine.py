from vedacore.misc import registry
from .infer_engine import InferEngine


@registry.register_module('engine')
class ValEngine(InferEngine):

    def __init__(self, model, meshgrid, converter, num_classes, use_sigmoid,
                 test_cfg):
        super().__init__(model, meshgrid, converter, num_classes, use_sigmoid,
                         test_cfg)

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, imgs, video_metas):
        dets = self.infer(imgs, video_metas)
        return dets
