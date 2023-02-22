from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation, hard_and, hard_or, initialization


def soft_mask_to_true_deprecated(w: float, x: float):
    """
    w > 0.5 implies the mask operation is inactive, else active

    Assumes x is in [0, 1]

    Corresponding hard logic: x OR ! w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return jax.numpy.maximum(x, 1.0 - w)

# Superior on noisy XOR
def soft_mask_to_true(w: float, x: float) -> float:
    w = jax.numpy.clip(w, 0.0, 1.0)
    return hard_or.soft_or(x, 1.0 - w)



def hard_mask_to_true(w, x):
    return jax.numpy.logical_or(x, jax.numpy.logical_not(w))


soft_mask_to_true_neuron = jax.vmap(soft_mask_to_true, 0, 0)

hard_mask_to_true_neuron = jax.vmap(hard_mask_to_true, 0, 0)


soft_mask_to_true_layer = jax.vmap(soft_mask_to_true_neuron, (0, None), 0)

hard_mask_to_true_layer = jax.vmap(hard_mask_to_true_neuron, (0, None), 0)


def soft_mask_to_false_deprecated(w: float, x: float):
    """
    w > 0.5 implies the mask is inactive, else active

    Assumes x is in [0, 1]

    Corresponding hard logic: b AND w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    # TODO: what is this madness?
    return 1.0 - jax.numpy.maximum(1.0 - x, 1.0 - w)

# Superior on noisy XOR
def soft_mask_to_false(w: float, x: float) -> float:
    w = jax.numpy.clip(w, 0.0, 1.0)
    return hard_and.soft_and(x, w)


def hard_mask_to_false(w, x):
    return jax.numpy.logical_and(x, w)


soft_mask_to_false_neuron = jax.vmap(soft_mask_to_false, 0, 0)

hard_mask_to_false_neuron = jax.vmap(hard_mask_to_false, 0, 0)


soft_mask_to_false_layer = jax.vmap(soft_mask_to_false_neuron, (0, None), 0)

hard_mask_to_false_layer = jax.vmap(hard_mask_to_false_neuron, (0, None), 0)


class SoftMaskLayer(nn.Module):
    mask_layer_operation: Callable
    layer_size: int
    weights_init: Callable = nn.initializers.uniform(1.0)
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param(
            "bit_weights", self.weights_init, weights_shape, self.dtype
        )
        x = jax.numpy.asarray(x, self.dtype)
        return self.mask_layer_operation(weights, x)


class HardMaskLayer(nn.Module):
    mask_layer_operation: Callable
    layer_size: int
    weights_init: Callable = nn.initializers.constant(True)

    @nn.compact
    def __call__(self, x):
        weights_shape = (self.layer_size, jax.numpy.shape(x)[-1])
        weights = self.param("bit_weights", self.weights_init, weights_shape)
        return self.mask_layer_operation(weights, x)


class SymbolicMaskLayer:
    def __init__(self, mask_layer):
        self.hard_mask_layer = mask_layer

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(self.hard_mask_layer, x)
        return symbolic_generation.symbolic_expression(jaxpr, x)


mask_to_true_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SoftMaskLayer(
        soft_mask_to_true_layer, layer_size, weights_init, dtype
    ),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: HardMaskLayer(hard_mask_to_true_layer, layer_size),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SymbolicMaskLayer(
        HardMaskLayer(hard_mask_to_true_layer, layer_size)
    ),
)


mask_to_false_layer = neural_logic_net.select(
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SoftMaskLayer(
        soft_mask_to_false_layer, layer_size, weights_init, dtype
    ),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: HardMaskLayer(hard_mask_to_false_layer, layer_size),
    lambda layer_size, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SymbolicMaskLayer(
        HardMaskLayer(hard_mask_to_false_layer, layer_size)
    ),
)
