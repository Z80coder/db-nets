from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation


def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is inactive, else active

    Assumes x is in [0, 1]

    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)


def hard_not(w: bool, x: bool) -> bool:
    return jax.numpy.logical_not(jax.numpy.logical_xor(x, w))


soft_not_neuron = jax.vmap(soft_not, 0, 0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)



soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)




class SoftNotLayer(nn.Module):
    layer_size: int
    weights_init: Callable = nn.initializers.uniform(1.0)
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', self.weights_init,
                             weights_shape, self.dtype)
        x = jax.numpy.asarray(x, self.dtype)
        return soft_not_layer(weights, x)


class HardNotLayer(nn.Module):
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            'weights', nn.initializers.constant(0.0), weights_shape)
        return hard_not_layer(weights, x)


class SymbolicNotLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_not_layer = HardNotLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_not_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


not_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0), dtype=jax.numpy.float32: SoftNotLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0), dtype=jax.numpy.float32: HardNotLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.uniform(1.0), dtype=jax.numpy.float32: SymbolicNotLayer(layer_size))
