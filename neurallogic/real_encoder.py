import jax


def soft_real_encoder(t: float, x: float) -> float:
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


def hard_real_encoder(t: float, x: float) -> bool:
    # t and x must be floats
    return jax.numpy.where(soft_real_encoder(t, x) > 0.5, True, False)


soft_real_encoder_neuron = jax.vmap(soft_real_encoder, in_axes=(0, None))

hard_real_encoder_neuron = jax.vmap(hard_real_encoder, in_axes=(0, None))

soft_real_encoder_layer = jax.vmap(soft_real_encoder_neuron, (0, 0), 0)

hard_real_encoder_layer = jax.vmap(hard_real_encoder_neuron, (0, 0), 0)


