from typing import Callable

import jax
from flax import linen as nn

from neurallogic import hard_masks, neural_logic_net, symbolic_generation


# TODO: seperate and operation from mask operation
def soft_and_neuron(w, x):
    x = jax.vmap(hard_masks.soft_mask_to_true, 0, 0)(w, x)
    return jax.numpy.min(x)


def hard_and_neuron(w, x):
    x = jax.vmap(hard_masks.hard_mask_to_true, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.lax.bitwise_and, [0])


soft_and_layer = jax.vmap(soft_and_neuron, (0, None), 0)

hard_and_layer = jax.vmap(hard_and_neuron, (0, None), 0)


# TODO: move initialization to separate file
# TODO: simplify initialization to avoid the need to specify a guassian mean and std
def initialize_near_to_zero(mean=-1, std=0.5):
    # TODO: investigate better initialization
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # Sample from standard normal distribution (zero mean, unit variance)
        x = jax.random.normal(key, shape, dtype)
        # Transform to a normal distribution with mean -1 and standard deviation 0.5
        x = std * x + mean
        x = jax.numpy.clip(x, 0.001, 0.999)
        return x

    return init


class SoftAndLayer(nn.Module):
    """
    A soft-bit AND layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """

    layer_size: int
    weights_init: Callable = initialize_near_to_zero()
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", self.weights_init, weights_shape, self.dtype
        )
        x = jax.numpy.asarray(x, self.dtype)
        return soft_and_layer(weights, x)


class HardAndLayer(nn.Module):
    """
    A hard-bit And layer that shadows the SoftAndLayer.
    This is a convenience class to make it easier to switch between soft and hard logic.

    Attributes:
        layer_size: The number of neurons in the layer.
    """

    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", nn.initializers.constant(True), weights_shape
        )
        return hard_and_layer(weights, x)


class SymbolicAndLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_and_layer = HardAndLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_and_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


and_layer = neural_logic_net.select(
    lambda layer_size, weights_init=initialize_near_to_zero(), dtype=jax.numpy.float32: SoftAndLayer(
        layer_size, weights_init, dtype
    ),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: HardAndLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: SymbolicAndLayer(layer_size),
)
