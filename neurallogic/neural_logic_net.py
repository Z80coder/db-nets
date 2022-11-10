from enum import Enum
from flax import linen as nn

NetType = Enum('NetType', ['Soft', 'Hard'])

def net(f):
    class SoftNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Soft, x)
    class HardNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Hard, x)
    return SoftNet(), HardNet()