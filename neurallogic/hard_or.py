from functools import reduce
from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net


def soft_or_include(w: float, x: float) -> float:
    """
    w > 0.5 implies the and operation is active, else inactive

    Assumes x is in [0, 1]
    
    Corresponding hard logic: b AND w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - jax.numpy.maximum(1.0 - x, 1.0 - w)

@jax.jit
def hard_or_include(w: bool, x: bool) -> bool:
    return x & w

def symbolic_or_include(w, x):
    expression = f"({x} and {w})"
    # Check if w is of type bool
    if isinstance(w, bool) and isinstance(x, bool):
        # We know the value of w and x, so we can evaluate the expression
        return eval(expression)
    # We don't know the value of w or x, so we return the expression
    return expression

def soft_or_neuron(w, x):
    x = jax.vmap(soft_or_include, 0, 0)(w, x)
    return jax.numpy.max(x)

def hard_or_neuron(w, x):
    x = jax.vmap(hard_or_include, 0, 0)(w, x)
    return jax.lax.reduce(x, False, jax.lax.bitwise_or, [0])

def symbolic_or_neuron(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    y = [symbolic_or_include(wi, xi) for wi, xi in zip(w, x)]
    expression = "(" + str(reduce(lambda a, b: f"{a} or {b}", y)) + ")"
    if all(isinstance(yi, bool) for yi in y):
        # We know the value of all yis, so we can evaluate the expression
        return eval(expression)
    return expression

soft_or_layer = jax.vmap(soft_or_neuron, (0, None), 0)

hard_or_layer = jax.vmap(hard_or_neuron, (0, None), 0)

def symbolic_or_layer(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    return [symbolic_or_neuron(wi, x) for wi in w]

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
    """
    A soft-bit Or layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """
    layer_size: int
    weights_init: Callable = initialize_near_to_one()
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', self.weights_init, weights_shape, self.dtype)
        x = jax.numpy.asarray(x, self.dtype)
        return soft_or_layer(weights, x)

class HardOrLayer(nn.Module):
    """
    A hard-bit Or layer that shadows the SoftAndLayer.
    This is a convenience class to make it easier to switch between soft and hard logic.

    Attributes:
        layer_size: The number of neurons in the layer.
    """
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', nn.initializers.constant(0.0), weights_shape)
        return hard_or_layer(weights, x)

class SymbolicOrLayer(nn.Module):
    """A symbolic Or layer than transforms its inputs along the last dimension.
    Attributes:
        layer_size: The number of neurons in the layer.
    """
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', nn.initializers.constant(0.0), weights_shape)
        weights = weights.tolist()
        if not isinstance(x, list):
            raise TypeError(f"Input {x} should be a list")
        return symbolic_or_layer(weights, x)

def or_layer(layer_size: int, type: neural_logic_net.NetType, weights_init: Callable = initialize_near_to_one(), dtype: jax.numpy.dtype = jax.numpy.float32):
    return neural_logic_net.select(SoftOrLayer(layer_size, weights_init, dtype), HardOrLayer(layer_size), SymbolicOrLayer(layer_size))(type)

