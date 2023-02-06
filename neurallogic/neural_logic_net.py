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
        def __call__(self, x, **kwargs):
            return f(NetType.Hard, x, **kwargs)

    class SymbolicNet(nn.Module):
        @nn.compact
        def __call__(self, x, **kwargs):
            return f(NetType.Symbolic, x, **kwargs)

    return SoftNet(), HardNet(), SymbolicNet()
