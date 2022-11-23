import jax.numpy as jnp
import numpy
import operator

from neurallogic import neural_logic_net

"""
    symbolic shape transformations
"""

def symbolic_ravel(x):
    return numpy.array(x).ravel().tolist()

nl_ravel = neural_logic_net.select(jnp.ravel, jnp.ravel, symbolic_ravel)

def symbolic_reshape(x, newshape):
    return jnp.array(x).reshape(newshape).tolist()

nl_reshape = neural_logic_net.select(jnp.reshape, jnp.reshape, symbolic_reshape)

"""
    symbolic computations
"""
def symbolic_reduce_impl(op, x, axis):
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

def symbolic_reduce(op, x, axis=None):
    if axis is None:
        # Special case for reducing all elements in a tensor
        while isinstance(x, list) and len(x) > 1:
            x = symbolic_reduce_impl(op, x, 0)
        return x
    else:
        x = symbolic_reduce_impl(op, x, axis)
    return x


nl_symbolic_sum = lambda x, axis=None: symbolic_reduce((operator.add, "+"), x, axis)

nl_sum = neural_logic_net.select(jnp.sum, jnp.sum, nl_symbolic_sum)

def nl_symbolic_mean(x, axis=None):
    n = numpy.prod(numpy.array(x).shape)       
    x = nl_symbolic_sum(x, axis)
    # TODO: implement symbolic division
    return x

nl_mean = neural_logic_net.select(jnp.mean, jnp.mean, nl_symbolic_mean)