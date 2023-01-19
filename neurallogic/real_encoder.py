from typing import Callable

import jax
from flax import linen as nn

from neurallogic import neural_logic_net, symbolic_generation

def soft_real_encoder(t: float, x: float) -> float:
    eps = 0.0000001
    # x should be in [0, 1]
    t = jax.numpy.clip(t, 0.0, 1.0)
    return jax.numpy.where(
        jax.numpy.isclose(t, x),
        0.5,
        # t != x
        jax.numpy.where(
            x < t,
            (1.0 / (2.0 * t + eps)) * x,
            # x > t
            (1.0 / (2.0 * (1.0 - t) + eps)) * (x + 1.0 - 2.0 * t)
        )
    )


def hard_real_encoder(t: float, x: float) -> bool:
    # t and x must be floats
    return jax.numpy.where(soft_real_encoder(t, x) > 0.5, True, False)


soft_real_encoder_neuron = jax.vmap(soft_real_encoder, in_axes=(0, None))

hard_real_encoder_neuron = jax.vmap(hard_real_encoder, in_axes=(0, None))

soft_real_encoder_layer = jax.vmap(soft_real_encoder_neuron, (0, 0), 0)

hard_real_encoder_layer = jax.vmap(hard_real_encoder_neuron, (0, 0), 0)


class SoftRealEncoderLayer(nn.Module):
    bits_per_real: int
    thresholds_init: Callable = nn.initializers.uniform(1.0)
    dtype: jax.numpy.dtype = jax.numpy.float32

    @nn.compact
    def __call__(self, x):
        thresholds_shape = (jax.numpy.shape(x)[-1], self.bits_per_real)
        thresholds = self.param("thresholds", self.thresholds_init, thresholds_shape, self.dtype)
        x = jax.numpy.asarray(x, self.dtype)
        return soft_real_encoder_layer(thresholds, x)


class HardRealEncoderLayer(nn.Module):
    bits_per_real: int

    @nn.compact
    def __call__(self, x):
        thresholds_shape = (jax.numpy.shape(x)[-1], self.bits_per_real)
        thresholds = self.param("thresholds", nn.initializers.constant(0.0), thresholds_shape)
        return hard_real_encoder_layer(thresholds, x)


class SymbolicRealEncoderLayer:
    def __init__(self, bits_per_real):
        self.bits_per_real = bits_per_real
        self.hard_real_encoder_layer = HardRealEncoderLayer(self.bits_per_real)

    def __call__(self, x):
        jaxpr = symbolic_generation.make_symbolic_flax_jaxpr(
            self.hard_real_encoder_layer, x
        )
        return symbolic_generation.symbolic_expression(jaxpr, x)


real_encoder_layer = neural_logic_net.select(
    lambda bits_per_real, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SoftRealEncoderLayer(
        bits_per_real, weights_init, dtype
    ),
    lambda bits_per_real, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: HardRealEncoderLayer(bits_per_real),
    lambda bits_per_real, weights_init=nn.initializers.uniform(
        1.0
    ), dtype=jax.numpy.float32: SymbolicRealEncoderLayer(bits_per_real),
)
