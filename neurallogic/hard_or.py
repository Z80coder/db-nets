from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation, hard_masks


# TODO: seperate out the or operation from the mask operation
def soft_or_neuron(w, x):
    x = jax.vmap(hard_masks.soft_mask_to_false, 0, 0)(w, x)
    return jax.numpy.max(x)

def soft_or(x, y):
    m = jax.numpy.maximum(x, y)
    return jax.numpy.where(
        2 * m > 1,
        0.5 + 0.5 * (x + y) * (m - 0.5),
        m + 0.5 * (x + y) * (0.5 - m),
    )


def soft_or_neuron_deprecated(w, x):
    x = jax.vmap(hard_masks.soft_mask_to_true, 0, 0)(w, x)
    return jax.lax.reduce(x, 0.0, soft_or, [0])

def hard_or_neuron(w, x):
    x = jax.vmap(hard_masks.hard_mask_to_false, 0, 0)(w, x)
    return jax.lax.reduce(x, False, jax.lax.bitwise_or, [0])


soft_or_layer = jax.vmap(soft_or_neuron, (0, None), 0)

hard_or_layer = jax.vmap(hard_or_neuron, (0, None), 0)

# TODO: investigate better initialization


def initialize_near_to_one():
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # Sample from standard normal distribution (zero mean, unit variance)
        x = jax.random.normal(key, shape, dtype)
        # Transform to a normal distribution with mean 1 and standard deviation 0.5
        x = 0.5 * x + 1
        x = jax.numpy.clip(x, 0.001, 0.999)
        return x

    return init


class SoftOrLayer(nn.Module):
    layer_size: int
    weights_init: Callable = initialize_near_to_one()
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", self.weights_init, weights_shape, self.dtype
        )
        x = jax.numpy.asarray(x, self.dtype)
        return soft_or_layer(weights, x)


class HardOrLayer(nn.Module):
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", nn.initializers.constant(True), weights_shape
        )
        return hard_or_layer(weights, x)


class SymbolicOrLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_or_layer = HardOrLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_or_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


or_layer = neural_logic_net.select(
    lambda layer_size, weights_init=initialize_near_to_one(), dtype=jax.numpy.float32: SoftOrLayer(
        layer_size, weights_init, dtype
    ),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: HardOrLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: SymbolicOrLayer(layer_size),
)
