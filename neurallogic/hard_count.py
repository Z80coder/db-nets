import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation


def high_to_low(x, y):
    return jax.numpy.minimum(1 - x, y)

def soft_count(x: jax.numpy.array):
    """
    Returns an array of soft-bits, of length |x|+1, and where only 1 soft-bit is high.
    The index of the high soft-bit indicates the total quantity of low and high bits in the input array.
    i.e. if index i is high, then there are i low bits

    E.g. if x = [0.1, 0.9, 0.2, 0.6, 0.4], then the output is y=[low, low, low, high, low, low]
    y[3] is high, which indicates that
        - 3 bits are low
        - 2 bits are high

    E.g. if x = [0.0, 0.2, 0.3, 0.1, 0.4], then the output is y=[low, low, low, low, low, high]
    y[5] is high, which indicates that
        - 5 bits are low
        - 0 bits are high
    
    E.g. if x = [0.9, 0.8, 0.7, 0.6, 0.5], then the output is y=[high, low, low, low, low, low]
    y[0] is high, which indicates that
        - 0 bits are low
        - 5 bits are high
    """
    sorted_x = jax.numpy.sort(x, axis=-1)
    low = jax.numpy.array([0.0])
    high = jax.numpy.array([1.0])
    sorted_x = jax.numpy.concatenate([low, sorted_x, high])
    return jax.vmap(high_to_low)(sorted_x[:-1], sorted_x[1:])
    
def hard_count(x: jax.numpy.array):
    # We simply count the number of low bits
    num_low_bits = jax.numpy.sum(x <= 0.5, axis=-1)
    return jax.nn.one_hot(num_low_bits, num_classes=x.shape[-1] + 1)
    

soft_count_layer = jax.vmap(soft_count, in_axes=0)

hard_count_layer = jax.vmap(hard_count, in_axes=0)


class SoftCountLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return soft_count_layer(x)


class HardCountLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return hard_count_layer(x)


class SymbolicCountLayer:
    def __init__(self):
        self.hard_count_layer = HardCountLayer()

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(
            self.hard_count_layer, x
        )
        return symbolic_generation.symbolic_expression(jaxpr, x)


count_layer = neural_logic_net.select(
    lambda: SoftCountLayer(),
    lambda: HardCountLayer(),
    lambda: SymbolicCountLayer(),
)

