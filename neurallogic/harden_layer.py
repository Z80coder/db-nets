import jax

from neurallogic import neural_logic_net


def logistic_clip(x):
    return jax.scipy.special.expit(3 * (2 * x - 1))


def harden(x):
    # non-differentiable
    return jax.lax.cond(x > 0.5, lambda _: 1.0, lambda _: 0.0, None)


def straight_through_harden(x):
    # The harden operation is non-differentiable. Therefore we need to
    # approximate with the straight-through estimator.

    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = x - jax.lax.stop_gradient(x)
    grad_of_one = zero + jax.lax.stop_gradient(harden(x))
    return grad_of_one


soft_harden_layer = jax.vmap(straight_through_harden)


def hard_harden_layer(x):
    return x


# TODO: can we harden arbitrary tensors?
# TODO: is this correct?
def symbolic_harden_layer(x):
    return x


harden_layer = neural_logic_net.select(
    soft_harden_layer, hard_harden_layer, symbolic_harden_layer
)
 