from typing import Any
from functools import reduce
from typing import (Callable, Mapping)

import numpy
import jax
from flax import linen as nn

from neurallogic import neural_logic_net, sym_gen, symbolic_primitives


def soft_and_include(w: float, x: float) -> float:
    """
    w > 0.5 implies the and operation is active, else inactive

    Assumes x is in [0, 1]

    Corresponding hard logic: x OR ! w
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return jax.numpy.maximum(x, 1.0 - w)



def hard_and_include(w, x):
    return jax.numpy.logical_or(x, jax.numpy.logical_not(w))



def soft_and_neuron(w, x):
    x = jax.vmap(soft_and_include, 0, 0)(w, x)
    return jax.numpy.min(x)


def hard_and_neuron(w, x):
    x = jax.vmap(hard_and_include, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.lax.bitwise_and, [0])


soft_and_layer = jax.vmap(soft_and_neuron, (0, None), 0)

hard_and_layer = jax.vmap(hard_and_neuron, (0, None), 0)



def initialize_near_to_zero():
    # TODO: investigate better initialization
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # Sample from standard normal distribution (zero mean, unit variance)
        x = jax.random.normal(key, shape, dtype)
        # Transform to a normal distribution with mean -1 and standard deviation 0.5
        x = 0.5 * x - 1
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
        weights = self.param('weights', self.weights_init,
                             weights_shape, self.dtype)
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
            'weights', nn.initializers.constant(0.0), weights_shape)
        return hard_and_layer(weights, x)


class JaxprAndLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_and_layer = HardAndLayer(self.layer_size)

    def __call__(self, x):
        jaxpr = sym_gen.make_symbolic_jaxpr(self.hard_and_layer, x)
        return sym_gen.eval_symbolic(jaxpr, x)


def my_scope_put_variable(self, col: str, name: str, value: Any):
    self._check_valid()
    self._validate_trace_level()
    variables = self._collection(col)

    def put(target, key, val):
        if (key in target and isinstance(target[key], dict) and
                isinstance(val, Mapping)):
            for k, v in val.items():
                put(target[key], k, v)
        else:
            target[key] = val

    put(variables, name, value)


def my_put_variable(self, col: str, name: str, value: Any):
    if self.scope is None:
        raise ValueError("Can't access variables on unbound modules")
    self.scope._variables = self.scope.variables().unfreeze()
    my_scope_put_variable(self.scope, col, name, value)


class SymbolicAndLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_and_layer = HardAndLayer(self.layer_size)

    def __call__(self, x):
        symbolic_weights = self.hard_and_layer.get_variable("params", "weights")
        if isinstance(symbolic_weights, list) or (isinstance(symbolic_weights, numpy.ndarray) and symbolic_weights.dtype == object):
            symbolic_weights_n = symbolic_primitives.map_at_elements(symbolic_weights, lambda x: 0)
            symbolic_weights_n = numpy.asarray(symbolic_weights_n, dtype=numpy.float32)
            my_put_variable(self.hard_and_layer, "params", "weights", symbolic_weights_n)
        if isinstance(x, list) or (isinstance(x, numpy.ndarray) and x.dtype == object):
            xn = symbolic_primitives.map_at_elements(x, lambda x: 0)
            xn = numpy.asarray(xn, dtype=numpy.float32)
        else:
            xn = x
        jaxpr = sym_gen.make_symbolic_jaxpr(self.hard_and_layer, xn)
        # Swap out the numeric consts (that represent the weights) for the symbolic weights
        jaxpr.consts = [symbolic_weights]
        return sym_gen.symbolic_expression(jaxpr, x)


and_layer = neural_logic_net.select(
    lambda layer_size, weights_init=initialize_near_to_zero(),
    dtype=jax.numpy.float32: SoftAndLayer(layer_size, weights_init, dtype),
    lambda layer_size, weights_init=initialize_near_to_zero(
    ), dtype=jax.numpy.float32: HardAndLayer(layer_size),
    lambda layer_size, weights_init=initialize_near_to_zero(
    ), dtype=jax.numpy.float32: JaxprAndLayer(layer_size),
    lambda layer_size, weights_init=initialize_near_to_zero(),
    dtype=jax.numpy.float32: SymbolicAndLayer(layer_size))
