from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation, hard_masks


def differentiable_xor(x, y):
    return jax.numpy.minimum(jax.numpy.maximum(x, y), 1.0 - jax.numpy.minimum(x, y))


# TODO: seperate out the mask from the xor operation
def soft_xor_neuron(w, x):
    # Conditionally include input bits, according to weights
    x = jax.vmap(hard_masks.soft_mask_to_false, 0, 0)(w, x)
    x = jax.lax.reduce(x, jax.numpy.array(0, dtype=x.dtype), differentiable_xor, (0,))
    return x


def hard_xor_neuron(w, x):
    x = jax.vmap(hard_masks.hard_mask_to_false, 0, 0)(w, x)
    return jax.lax.reduce(x, False, jax.lax.bitwise_xor, [0])


soft_xor_layer = jax.vmap(soft_xor_neuron, (0, None), 0)


hard_xor_layer = jax.vmap(hard_xor_neuron, (0, None), 0)


class SoftXorLayer(nn.Module):
    layer_size: int
    weights_init: Callable = (
        nn.initializers.uniform(1.0)
    )
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", self.weights_init, weights_shape, self.dtype
        )
        x = jax.numpy.asarray(x, self.dtype)
        return soft_xor_layer(weights, x)


class HardXorLayer(nn.Module):
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", nn.initializers.constant(True), weights_shape
        )
        return hard_xor_layer(weights, x)


class SymbolicXorLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_xor_layer = HardXorLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_xor_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


xor_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SoftXorLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: HardXorLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: SymbolicXorLayer(layer_size),
)
