import jax
from jax import lax
import numpy

# TODO: clip weights between 0 and 1
def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is active, else inactive
    
    Corresponding hard logic: 
        (x AND w) || (! x AND ! w) 
    or equivalently 
        ! (x XOR w)
    """
    return 1.0 - w + x * (2.0 * w - 1.0)

@jax.jit
def hard_not(w, x):
    return (x & w) | (~x & ~w)

soft_not_neuron = jax.vmap(soft_not, 0, 0)
hard_not_neuron = jax.vmap(hard_not, 0, 0)
soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)
hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)

def hard_not_deprecated(w, x):
    return numpy.logical_not(numpy.logical_xor(w, x))
def hard_not_neuron_deprecated(w, x): return hard_not(w, x)
