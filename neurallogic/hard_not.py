import jax

def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is active, else inactive

    Assumes x is in [0, 1]
    
    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)

@jax.jit
def hard_not(w: bool, x: bool) -> bool:
    return ~(x ^ w)

soft_not_neuron = jax.vmap(soft_not, 0, 0)
hard_not_neuron = jax.vmap(hard_not, 0, 0)
soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)
hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)

def not_layer():
    return soft_not_layer, hard_not_layer