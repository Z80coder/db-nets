import jax.numpy as jnp

from neurallogic import neural_logic_net

def symbolic_ravel(x):
    if isinstance(x, list):
        return [item for sublist in x for item in symbolic_ravel(sublist)]
    else:
        return [x]

nl_ravel = neural_logic_net.select(jnp.ravel, jnp.ravel, symbolic_ravel)

def symbolic_sum(x):
    if isinstance(x, list):
        return sum([symbolic_sum(sublist) for sublist in x])
    else:
        return x

nl_sum = neural_logic_net.select(jnp.sum, jnp.sum, symbolic_sum)

def symbolic_reshape(x, newshape):
    return jnp.array(x).reshape(newshape).tolist()

nl_reshape = neural_logic_net.select(jnp.reshape, jnp.reshape, symbolic_reshape)

