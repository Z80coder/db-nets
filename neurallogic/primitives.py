from neurallogic import neural_logic_net

import jax.numpy as jnp

def symbolic_ravel(x):
    if isinstance(x, list):
        return [item for sublist in x for item in symbolic_ravel(sublist)]
    else:
        return [x]

def ravel(type: neural_logic_net.NetType):
    return {
        neural_logic_net.NetType.Soft: jnp.ravel,
        neural_logic_net.NetType.Hard: jnp.ravel,
        neural_logic_net.NetType.Symbolic: symbolic_ravel
    }[type]
