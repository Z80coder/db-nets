import jax


def soft_bit(t: float, x: float) -> float:
    # x should be in [0, 1]
    t = jax.numpy.clip(t, 0.0, 1.0)
    return jax.numpy.where(
        x == t,
        0.5,
        jax.numpy.where(
            x < t,
            (1.0 / (2.0 * t)) * x,
            (1.0 / (2.0 * (1.0 - t))) * (x + 1.0 - 2.0 * t),
        ),
    )


def hard_bit(t: float, x: float) -> bool:
    # t and x must be floats
    return jax.numpy.where(soft_bit(t, x) > 0.5, True, False)


soft_bit_neuron = jax.vmap(soft_bit, in_axes=(0, None))

hard_bit_neuron = jax.vmap(hard_bit, in_axes=(0, None))

soft_bit_layer = jax.vmap(soft_bit_neuron, (0, 0), 0)

hard_bit_layer = jax.vmap(hard_bit_neuron, (0, 0), 0)
