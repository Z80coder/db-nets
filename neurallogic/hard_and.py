from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net

jax.lax.create_token

def soft_and_include(w: float, x: float) -> float:
    """
    w > 0.5 implies the and operation is active, else inactive

    Assumes x is in [0, 1]
    
    Corresponding hard logic: x OR ! w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return jax.numpy.maximum(x, 1.0 - w)

# TODO: why do I need to jax.jit this?
@jax.jit
def hard_and_include(w: bool, x: bool) -> bool:
    return x | ~w

def symbolic_and_include(w, x):
    return eval(f"{x} or not({w})")

def soft_and_neuron(w, x):
    x = jax.vmap(soft_and_include, 0, 0)(w, x)
    return jax.numpy.min(x)

def hard_and_neuron(w, x):
    x = jax.vmap(hard_and_include, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.lax.bitwise_and, [0])

def symbolic_and_neuron(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    y = [symbolic_and_include(wi, xi) for wi, xi in zip(w, x)]
    return all(y)

soft_and_layer = jax.vmap(soft_and_neuron, (0, None), 0)

hard_and_layer = jax.vmap(hard_and_neuron, (0, None), 0)

def symbolic_and_layer(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    return [symbolic_and_neuron(wi, x) for wi in w]

class SoftAndLayer(nn.Module):
    """
    A soft-bit AND layer than transforms its inputs along the last dimension.

    Attributes:
        layer_size: The number of neurons in the layer.
        weights_init: The initializer function for the weight matrix.
    """
    layer_size: int
    weights_init: Callable = nn.initializers.uniform(1.0)

    @nn.compact
    def __call__(self, x):
        dtype = jax.numpy.float32
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', self.weights_init, weights_shape, dtype)
        # TODO: do we need this?
        x = jax.numpy.asarray(x, dtype)
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
        weights = self.param('weights', nn.initializers.constant(0.0), weights_shape)
        return hard_and_layer(weights, x)

class SymbolicAndLayer(nn.Module):
    """A symbolic And layer than transforms its inputs along the last dimension.
    Attributes:
        layer_size: The number of neurons in the layer.
    """
    layer_size: int

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param('weights', nn.initializers.constant(0.0), weights_shape)
        return symbolic_and_layer(weights, x)

def AndLayer(layer_size: int, type: neural_logic_net.NetType) -> nn.Module:
    return {
        neural_logic_net.NetType.Soft: SoftAndLayer(layer_size),
        neural_logic_net.NetType.Hard: HardAndLayer(layer_size),
        neural_logic_net.NetType.Symbolic: SymbolicAndLayer(layer_size)
    }[type]

