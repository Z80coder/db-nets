from enum import Enum
from flax import linen as nn

NetType = Enum('NetType', ['Soft', 'Hard', 'Jaxpr', 'Symbolic'])

def select(soft, hard, jaxpr, symbolic):
    def selector(type: NetType):
        return {
            NetType.Soft: soft,
            NetType.Hard: hard,
            NetType.Jaxpr: jaxpr,
            NetType.Symbolic: symbolic
        }[type]
    return selector

def net(f):
    class SoftNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Soft, x)
    class HardNet(nn.Module): 
        @nn.compact
        def __call__(self, x):
            return f(NetType.Hard, x)
    class JaxprNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Jaxpr, x)
    class SymbolicNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return f(NetType.Symbolic, x)
    return SoftNet(), HardNet(), JaxprNet(), SymbolicNet()

# TODO: support init of all three net types at once