import jax.numpy as jnp
import numpy
import operator

from neurallogic import neural_logic_net

"""
    symbolic shape transformations
"""

def symbolic_ravel_deprecated(x):
    return numpy.array(x).ravel().tolist()

nl_ravel_deprecated = neural_logic_net.select(jnp.ravel, jnp.ravel, symbolic_ravel_deprecated)

def symbolic_reshape_deprecated(x, newshape):
    return numpy.array(x).reshape(newshape).tolist()

nl_reshape_deprecated = neural_logic_net.select(lambda newshape: lambda x: jnp.reshape(x, newshape), lambda newshape: lambda x: jnp.reshape(x, newshape), lambda newshape: lambda x: symbolic_reshape_deprecated(x, newshape))

"""
    symbolic computations
"""
def symbolic_reduce_impl_deprecated(op, x, axis):
    """
        Cannot support multiple axes due to limitations of numpy.        
    """
    def op_xy(x, y):
        if isinstance(x, str) or isinstance(y, str):
            return f"({x}) {op[1]} ({y})"
        else:
            return op[0](x, y)
    f = numpy.frompyfunc(lambda x, y: op_xy(x,y), 2, 1)
    x = f.reduce(x, axis=axis)
    if isinstance(x, numpy.ndarray):
        x = x.tolist()
    return x

def symbolic_reduce_deprecated(op, x, axis=None):
    if axis is None:
        # Special case for reducing all elements in a tensor
        while isinstance(x, list) and len(x) > 1:
            x = symbolic_reduce_impl_deprecated(op, x, 0)
        return x
    else:
        x = symbolic_reduce_impl_deprecated(op, x, axis)
    return x

def symbolic_sum_deprecated(x, axis=None):
    return symbolic_reduce_deprecated((operator.add, "+"), x, axis)

nl_sum_deprecated = neural_logic_net.select(lambda axis=None: lambda x: jnp.sum(x, axis), lambda axis=None: lambda x: jnp.sum(x, axis), lambda axis=None: lambda x: symbolic_sum_deprecated(x, axis))
