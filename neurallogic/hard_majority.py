import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation


def majority_index(input_size: int) -> int:
    return (input_size - 1) // 2

# TODO: properly factor with/without margin versions

def majority_bit(x: jax.numpy.array) -> float:
    index = majority_index(x.shape[-1])
    sorted_x = jax.numpy.sort(x, axis=-1)
    return jax.numpy.take(sorted_x, index, axis=-1)


def soft_majority(x: jax.numpy.array) -> float:
    m_bit = majority_bit(x)
    margin = jax.numpy.abs(m_bit - 0.5)
    mean = jax.numpy.mean(x, axis=-1)
    margin_delta = mean * margin
    representative_bit = jax.numpy.where(
        m_bit > 0.5,
        0.5 + margin_delta,
        m_bit + margin_delta,
    )
    return representative_bit


def hard_majority(x: jax.numpy.array) -> bool:
    threshold = x.shape[-1] - majority_index(x.shape[-1])
    return jax.numpy.sum(x, axis=-1) >= threshold


soft_majority_layer = jax.vmap(soft_majority, in_axes=0)

hard_majority_layer = jax.vmap(hard_majority, in_axes=0)


class SoftMajorityLayer(nn.Module):
    """
    A soft-bit MAJORITY layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """

    @nn.compact
    def __call__(self, x):
        return soft_majority_layer(x)


class HardMajorityLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return hard_majority_layer(x)


class SymbolicMajorityLayer:
    def __init__(self):
        self.hard_majority_layer = HardMajorityLayer()

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(
            self.hard_majority_layer, x
        )
        return symbolic_generation.symbolic_expression(jaxpr, x)


majority_layer = neural_logic_net.select(
    lambda: SoftMajorityLayer(),
    lambda: HardMajorityLayer(),
    lambda: SymbolicMajorityLayer(),
)

# TODO: construct a majority-k generalisation of the above
# where k is the number of high-soft bits required for a majority
# and where k is a soft-bit parameter. Requires constructing
# a piecewise-continuous function (as per notebook).
