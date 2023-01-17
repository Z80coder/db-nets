from typing import Any
from functools import reduce
from typing import (Callable, Mapping)

import numpy
import jax
from flax import errors
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


# TODO: do we need to jit here? should apply jit at the highest level of the architecture
# TODO: may need to jit in unit tests, however
"""
@jax.jit
def hard_and_include(w: bool, x: bool) -> bool:
    print(f"hard_and_include: w={w}, x={x}")
    # TODO: this works when the function is jitted, but not when it is not jitted
    return x | ~w
    #return x or not w
"""


def hard_and_include(w, x):
    return jax.numpy.logical_or(x, jax.numpy.logical_not(w))

# def hard_and_include(w, x):
#    return jax.numpy.logical_or(x, jax.numpy.logical_not(w))


"""
def symbolic_and_include(w, x):
    expression = f"({x} or not({w}))"
    # Check if w is of type bool
    if isinstance(w, bool) and isinstance(x, bool):
        # We know the value of w and x, so we can evaluate the expression
        return eval(expression)
    # We don't know the value of w or x, so we return the expression
    return expression
"""

# def symbolic_and_include(w, x):
#    symbolic_f = sym_gen.make_symbolic(hard_and_include, w, x)
#    return sym_gen.eval_symbolic(symbolic_f, w, x)


def soft_and_neuron(w, x):
    x = jax.vmap(soft_and_include, 0, 0)(w, x)
    return jax.numpy.min(x)


def hard_and_neuron(w, x):
    x = jax.vmap(hard_and_include, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.lax.bitwise_and, [0])


"""
def hard_and_neuron(w, x):
    x = jax.vmap(hard_and_include, 0, 0)(w, x)
    return jax.lax.reduce(x, True, jax.numpy.logical_and, [0])
"""

"""
def symbolic_and_neuron(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    y = [symbolic_and_include(wi, xi) for wi, xi in zip(w, x)]
    expression = "(" + str(reduce(lambda a, b: f"{a} and {b}", y)) + ")"
    if all(isinstance(yi, bool) for yi in y):
        # We know the value of all yis, so we can evaluate the expression
        return eval(expression)
    return expression
"""

soft_and_layer = jax.vmap(soft_and_neuron, (0, None), 0)

hard_and_layer = jax.vmap(hard_and_neuron, (0, None), 0)

"""
def symbolic_and_layer(w, x):
    # TODO: ensure that this implementation has the same generality over tensors as vmap
    if not isinstance(w, list):
        raise TypeError(f"Input {x} should be a list")
    if not isinstance(x, list):
        raise TypeError(f"Input {x} should be a list")
    return [symbolic_and_neuron(wi, x) for wi in w]
"""

# def symbolic_and_layer(w, x):
#    symbolic_hard_and_layer = sym_gen.make_symbolic(hard_and_layer)
#    return sym_gen.eval_symbolic(symbolic_hard_and_layer, w, x)

# TODO: investigate better initialization


def initialize_near_to_zero():
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
    """Updates the value of the given variable if it is mutable, or an error otherwise.

    Args:
      col: the collection of the variable.
      name: the name of the variable.
      value: the new value of the given variable.
    """
    self._check_valid()
    self._validate_trace_level()
    #if not self.is_mutable_collection(col):
    #    raise errors.ModifyScopeVariableError(col, name, self.path_text)
    #variables = self._mutable_collection(col)
    variables = self._collection(col)
    # Make sure reference sharing of child variable dictionaries isn't broken

    def put(target, key, val):
        if (key in target and isinstance(target[key], dict) and
                isinstance(val, Mapping)):
            for k, v in val.items():
                put(target[key], k, v)
        else:
            target[key] = val

    put(variables, name, value)


def my_put_variable(self, col: str, name: str, value: Any):
    """Sets the value of a Variable.

    Args:
        col: the variable collection.
        name: the name of the variable.
        value: the new value of the variable.

    Returns:

    """
    if self.scope is None:
        raise ValueError("Can't access variables on unbound modules")
    mutable_variables = self.scope.variables().unfreeze()
    self.scope._variables = mutable_variables
    #mutated_self = my_scope_put_variable(self.scope, col, name, value)
    #self.scope.put_variable(col, name, value)
    my_scope_put_variable(self.scope, col, name, value)
    #immutable_scope = self.scope.variables.freeze()
    #return immutable_scope


class SymbolicAndLayer:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.hard_and_layer = HardAndLayer(self.layer_size)

    def __call__(self, x):
        symbolic_weights = self.hard_and_layer.get_variable("params", "weights")
        print(f'symbolic_weights: {symbolic_weights} of type {type(symbolic_weights)}')
        if isinstance(symbolic_weights, list) or (isinstance(symbolic_weights, numpy.ndarray) and symbolic_weights.dtype == numpy.object):
            symbolic_weights_n = symbolic_primitives.map_at_elements(symbolic_weights, lambda x: 0)
            symbolic_weights_n = numpy.asarray(symbolic_weights_n, dtype=numpy.float32)
            my_put_variable(self.hard_and_layer, "params", "weights", symbolic_weights_n)
            print(f'converted to symbolic_weights_n: {symbolic_weights_n} of type {type(symbolic_weights_n)}')

        #print(
        #    f'symbolic_weights: {symbolic_weights} of type {type(symbolic_weights)}')
        # Convert the symbolic inputs to numeric inputs so that we can generate a jaxpr
        #numeric_weights = sym_gen.make_numeric(symbolic_weights)
        #print(
        #    f'numeric_weights: {numeric_weights} of type {type(numeric_weights)}')
        #numeric_input = numpy.array(
        #    sym_gen.make_numeric(x), dtype=numpy.float32)
        #print(f'numeric_input: {numeric_input} of type {type(numeric_input)}')
        # Overwrite the supplied weights with the temporary numeric weights
        #my_put_variable(self.hard_and_layer, "params", "weights", symbolic_weights_n)
        # Generate the jaxpr for this layer
        #jaxpr = sym_gen.make_symbolic_jaxpr(self.hard_and_layer, numeric_input)
        print(f'x: {x} of type {type(x)}')
        if isinstance(x, numpy.ndarray):
            print(f'x is a numpy array with dtype = {x.dtype}')
        if isinstance(x, jax.numpy.ndarray):
            print(f'x is a jax.numpy array with dtype = {x.dtype}')
        #xn = sym_gen.make_numeric(x)
        #print(f'converted to xn: {xn} of type {type(xn)}')
        if isinstance(x, list) or (isinstance(x, numpy.ndarray) and x.dtype == numpy.object):
            # x = numpy.zeros_like(list)
            #x = numpy.asarray(x, dtype=numpy.float32)
            xn = symbolic_primitives.map_at_elements(x, lambda x: 0)
            xn = numpy.asarray(xn, dtype=numpy.float32)
        else:
            xn = x
        print(f'converted to xn: {xn} of type {type(xn)}')
        jaxpr = sym_gen.make_symbolic_jaxpr(self.hard_and_layer, xn)
        print(f'jaxpr consts: {jaxpr.consts} of type {type(jaxpr.consts)}')
        print(f'jaxpr: {jaxpr}')
        # Swap out the numeric consts (that represent the weights) for the symbolic weights
        jaxpr.consts = [symbolic_weights]
        #print(
        #    f'symbolic jaxpr consts: {jaxpr.consts} of type {type(jaxpr.consts)}')
        #symbolic_input = sym_gen.make_symbolic(x)
        #print(
        #    f'symbolic_input: {symbolic_input} of type {type(symbolic_input)}')
        print(f'x: {x} of type {type(x)}')
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
