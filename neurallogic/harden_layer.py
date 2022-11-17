import jax

from neurallogic import neural_logic_net


def harden_element(x):
    # non-differentiable
    return jax.lax.cond(x > 0.5, lambda _: 1.0, lambda _: 0.0, None)

def straight_through_harden_element(x):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(harden_element(x))

soft_harden_layer = jax.vmap(straight_through_harden_element)

def hard_harden_layer(x):
    return x

def symbolic_harden_layer(x):
    return x

harden_layer = neural_logic_net.select(soft_harden_layer, hard_harden_layer, symbolic_harden_layer)

