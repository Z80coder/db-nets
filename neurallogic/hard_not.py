import jax

def soft_not(w, x):
    """
    w > 0.5 implies the not operation is active, else inactive
    
    Corresponding hard logic: 
        (x AND w) || (! x AND ! w) 
    or equivalently 
        ! (x XOR w)
    """
    return 1.0 - w + x * (2.0 * w - 1.0)

soft_not_neuron = jax.vmap(soft_not, 0, 0)

soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)
