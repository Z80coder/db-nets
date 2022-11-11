from typing import Callable
import jax
from flax import linen as nn
from neurallogic import harden
from neurallogic import neural_logic_net

def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is inactive, else active

    Assumes x is in [0, 1]
    
    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)

# TODO: why do I need to jax.jit this?
@jax.jit
def hard_not(w: bool, x: bool) -> bool:
    return ~(x ^ w)

soft_not_neuron = jax.vmap(soft_not, 0, 0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)

soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)

class SoftNotLayer(nn.Module):
    """
    A soft-bit NOT layer than transforms its inputs along the last dimension.

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
        x = jax.numpy.asarray(x, dtype)
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
        weights = self.param('weights', nn.initializers.constant(0.0), weights_shape)
        return hard_not_layer(weights, x)

def NotLayer(layer_size: int, type: neural_logic_net.NetType) -> nn.Module:
    return {
        neural_logic_net.NetType.Soft: SoftNotLayer(layer_size),
        neural_logic_net.NetType.Hard: HardNotLayer(layer_size),
    }[type]

