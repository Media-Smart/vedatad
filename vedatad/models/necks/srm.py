import torch.nn as nn

from vedacore.misc import build_from_module, registry


@registry.register_module('neck')
class SRM(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRM, self).__init__()
        self.srm = build_from_module(srm_cfg, nn)

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.srm(x)
        x = x.squeeze(-1).squeeze(-1)

        return x
