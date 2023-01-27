from typing import Optional, Sequence

import jax
from flax import linen as nn
from jax import lax, random

from neurallogic import neural_logic_net


class SoftHardDropout(nn.Module):
    """Create a dropout layer suitable for dropping soft-bit values.
    Adapted from flax/stochastic.py


    Note: When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
    to include an RNG seed named `'dropout'`. For example::

      model.apply({'params': params}, inputs=inputs, train=True, rngs={'dropout': dropout_rng})`

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng_collection: the rng collection name to use when requesting an rng key.
    """

    rate: float
    broadcast_dims: Sequence[int] = ()
    deterministic: Optional[bool] = None
    rng_collection: str = "dropout"
    dropout_value: float = 0.0

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          Masking means setting the input bits to 0.5.
          deterministic: if false the inputs are masked,
          whereas if true, no mask is applied and the inputs are returned
          as is.

        Returns:
          The masked inputs
        """
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jax.numpy.zeros_like(inputs)

        keep_prob = 1.0 - self.rate
        rng = self.make_rng(self.rng_collection)
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jax.numpy.broadcast_to(mask, inputs.shape)
        masked_values = jax.numpy.full_like(inputs, self.dropout_value, dtype=float)
        return lax.select(mask, inputs, masked_values)


class HardHardDropout(nn.Module):
    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        return inputs


class SymbolicHardDropout(nn.Module):
    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        return inputs


hard_dropout = neural_logic_net.select(
    lambda **kwargs: SoftHardDropout(**kwargs),
    lambda **kwargs: HardHardDropout(**kwargs),
    lambda **kwargs: SymbolicHardDropout(**kwargs),
)
