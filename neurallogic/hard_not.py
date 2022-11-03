import jax

def hard_not(w, x):
    """
    w > 0.5 implies the not operation is active, else inactive
    The corresponding hard logic is: (x AND w) || (! x AND ! w) or equivalently ! (x XOR w)
    """
    return 1.0 - w + x * (2.0 * w - 1.0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)
