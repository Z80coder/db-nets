import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation


def soft_concatenate(x, axis):
    return jax.numpy.concatenate(x, axis)


def hard_concatenate(x, axis):
    return soft_concatenate(x, axis)


class SoftConcatenate(nn.Module):
    axis: int
    @nn.compact
    def __call__(self, x):
        return soft_concatenate(x, self.axis)


class HardConcatenate(nn.Module):
    axis: int
    @nn.compact
    def __call__(self, x):
        return hard_concatenate(x, self.axis)


class SymbolicConcatenate:
    def __init__(self, axis):
        self.hard_concatenate = HardConcatenate(axis)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(
            self.hard_concatenate, x
        )
        return symbolic_generation.symbolic_expression(jaxpr, x)


concatenate = neural_logic_net.select(
    lambda x, axis: SoftConcatenate(axis)(x),
    lambda x, axis: HardConcatenate(axis)(x),
    lambda x, axis: SymbolicConcatenate(axis)(x),
)

# TODO: add tests