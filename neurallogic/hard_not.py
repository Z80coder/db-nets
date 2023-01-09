from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net


def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is inactive, else active

    Assumes x is in [0, 1]

    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)


@jax.jit
def hard_not(w: bool, x: bool) -> bool:
    return ~(x ^ w)


def symbolic_not(w, x):
    expression = f"(not({x} ^ {w}))"
    # Check if w is of type bool
    if isinstance(w, bool) and isinstance(x, bool):
        # We know the value of w and x, so we can evaluate the expression
        return eval(expression)
    # We don't know the value of w or x, so we return the expression
    return expression


soft_not_neuron = jax.vmap(soft_not, 0, 0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)


def symbolic_not_neuron(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    return [symbolic_not(wi, xi) for wi, xi in zip(w, x)]


soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)


def symbolic_not_layer(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    return [symbolic_not_neuron(wi, x) for wi in w]


class SoftNotLayer(nn.Module):
    """
    A soft-bit NOT layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """
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
    """
    A hard-bit NOT layer that shadows the SoftNotLayer.
    This is a convenience class to make it easier to switch between soft and hard logic.

    Attributes:
        layer_size: The number of neurons in the layer.
    """
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            'weights', nn.initializers.constant(0.0), weights_shape)
        return hard_not_layer(weights, x)


class SymbolicNotLayer(nn.Module):
    """A symbolic NOT layer than transforms its inputs along the last dimension.
    Attributes:
        layer_size: The number of neurons in the layer.
    """
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            'weights', nn.initializers.constant(0.0), weights_shape)
        weights = weights.tolist()
        if not isinstance(x, list):
            raise TypeError(f"Input {x} should be a list")
        return symbolic_not_layer(weights, x)


not_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0), dtype=jax.numpy.float32: SoftNotLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0), dtype=jax.numpy.float32: HardNotLayer(layer_size),
    lambda layer_size, weights_init=nn.initializers.uniform(1.0), dtype=jax.numpy.float32: SymbolicNotLayer(layer_size))
