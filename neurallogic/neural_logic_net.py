from enum import Enum
from flax import linen as nn

NetType = Enum("NetType", ["Soft", "Hard", "Symbolic"])


def select(soft, hard, symbolic):
    def selector(type: NetType):
        return {NetType.Soft: soft, NetType.Hard: hard, NetType.Symbolic: symbolic}[
            type
        ]

    return selector


def net(f):
    class SoftNet(nn.Module):
        @nn.compact
        def __call__(self, x, **kwargs):
            return f(NetType.Soft, x, **kwargs)

    class HardNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Hard, x)

    class SymbolicNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Symbolic, x)

    return SoftNet(), HardNet(), SymbolicNet()
