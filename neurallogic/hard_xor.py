from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation


def soft_xor_include(w: float, x: float) -> float:
    """
    w > 0.5 implies the and operation is active, else inactive

    Assumes x is in [0, 1]

    Corresponding hard logic: b AND w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - jax.numpy.maximum(1.0 - x, 1.0 - w)


def hard_xor_include(w, x):
    return jax.numpy.logical_and(x, w)


def soft_xor_neuron(w, x):
    # Conditionally include input bits, according to weights
    x = jax.vmap(soft_xor_include, 0, 0)(w, x)
    # Compute the most sensitive bit
    margins = jax.vmap(lambda x: jax.numpy.abs(0.5 - x))(x)
    sensitive_bit_index = jax.numpy.argmin(margins)
    sensitive_bit = jax.numpy.take(x, sensitive_bit_index)
    # Compute the logical xor of the bits
    hard_x = jax.vmap(lambda x: jax.numpy.where(x > 0.5, True, False))(x)
    logical_xor = jax.lax.reduce(hard_x, False, jax.numpy.logical_xor, (0,))
    # Compute the representative bit
    hard_sensitive_bit = jax.numpy.where(sensitive_bit > 0.5, True, False)
    representative_bit = jax.numpy.where(logical_xor == hard_sensitive_bit,
                                         sensitive_bit,
                                         1.0 - sensitive_bit
                                         )
    return representative_bit


def hard_xor_neuron(w, x):
    x = jax.vmap(hard_xor_include, 0, 0)(w, x)
    return jax.lax.reduce(x, False, jax.lax.bitwise_xor, [0])


soft_xor_layer = jax.vmap(soft_xor_neuron, (0, None), 0)

hard_xor_layer = jax.vmap(hard_xor_neuron, (0, None), 0)


class SoftXorLayer(nn.Module):
    layer_size: int
    weights_init: Callable = nn.initializers.uniform(
        1.0)  # TODO: investigate better initialization
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            'bit_weights', self.weights_init, weights_shape, self.dtype)
        x = jax.numpy.asarray(x, self.dtype)
        return soft_xor_layer(weights, x)


class HardXorLayer(nn.Module):
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            'bit_weights', nn.initializers.constant(True), weights_shape)
        return hard_xor_layer(weights, x)


class SymbolicXorLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_xor_layer = HardXorLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(
            self.hard_xor_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


xor_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0), dtype=jax.numpy.float32: SoftXorLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=nn.initializers.constant(
        True), dtype=jax.numpy.float32: HardXorLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.constant(True), dtype=jax.numpy.float32: SymbolicXorLayer(layer_size))
