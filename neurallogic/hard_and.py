from typing import Callable

import jax
from flax import linen as nn

from neurallogic import hard_masks, neural_logic_net, symbolic_generation, initialization


# TODO: seperate and operation from mask operation
def soft_and_neuron(w, x):
    x = jax.vmap(hard_masks.soft_mask_to_true_margin, 0, 0)(w, x)
    return jax.numpy.min(x)


def soft_and(x, y):
    m = jax.numpy.minimum(x, y)
    return jax.numpy.where(
        2 * m > 1,
        0.5 + 0.5 * (x + y) * (m - 0.5),
        m + 0.5 * (x + y) * (0.5 - m),
    )

# This doesn't work well
def soft_and_neuron_deprecated(w, x):
    x = jax.vmap(hard_masks.soft_mask_to_true, 0, 0)(w, x)
    return jax.lax.reduce(x, 1.0, soft_and, [0])

def hard_and_neuron(w, x):
    x = jax.vmap(hard_masks.hard_mask_to_true, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.lax.bitwise_and, [0])


soft_and_layer = jax.vmap(soft_and_neuron, (0, None), 0)

hard_and_layer = jax.vmap(hard_and_neuron, (0, None), 0)



class SoftAndLayer(nn.Module):
    """
    A soft-bit AND layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """

    layer_size: int
    weights_init: Callable = initialization.initialize_near_to_zero()
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
    lambda layer_size, weights_init=initialization.initialize_near_to_zero(), dtype=jax.numpy.float32: SoftAndLayer(
        layer_size, weights_init, dtype
    ),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: HardAndLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.constant(
        True
    ), dtype=jax.numpy.float32: SymbolicAndLayer(layer_size),
)
