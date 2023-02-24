from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation, hard_and, hard_or, initialization


def soft_not(w, x):
    """
    w > 0.5 implies the not operation is inactive, else active

    Assumes x is in [0, 1]

    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)

# TODO: split out function of parameter, and not operation, in order to simplify
def soft_not_deprecated(w: float, x: float) -> float:
    w = jax.numpy.clip(w, 0.0, 1.0)
    # (w && x) || (! w && ! x)
    return hard_or.soft_or(hard_and.soft_and(w, x), hard_and.soft_and(1.0 - w, 1.0 - x))


def hard_not(w: bool, x: bool):
    return jax.numpy.logical_not(jax.numpy.logical_xor(x, w))


soft_not_neuron = jax.vmap(soft_not, 0, 0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)


soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)


class SoftNotLayer(nn.Module):
    layer_size: int
    weights_init: Callable = initialization.initialize_uniform_range(0.49, 0.51)
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", self.weights_init, weights_shape, self.dtype
        )
        x = jax.numpy.asarray(x, self.dtype)
        return soft_not_layer(weights, x)


class HardNotLayer(nn.Module):
    layer_size: int
    weights_init: Callable = nn.initializers.constant(True)

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param("bit_weights", self.weights_init, weights_shape)
        return hard_not_layer(weights, x)


class SymbolicNotLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_not_layer = HardNotLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_not_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


not_layer = neural_logic_net.select(
    lambda layer_size, weights_init=initialization.initialize_uniform_range(0.49, 0.51), dtype=jax.numpy.float32: SoftNotLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=initialization.initialize_uniform_range(0.49, 0.51), dtype=jax.numpy.float32: HardNotLayer(layer_size),
    lambda layer_size, weights_init=initialization.initialize_uniform_range(0.49, 0.51), dtype=jax.numpy.float32: SymbolicNotLayer(layer_size),
)
