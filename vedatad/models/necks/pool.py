import torch.nn as nn

from vedacore.misc import build_from_module, registry


@registry.register_module('neck')
class Pool(nn.Module):

    def __init__(self, pool):
        super(Pool, self).__init__()
        self.pool = build_from_module(pool, nn)

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)

        return x
