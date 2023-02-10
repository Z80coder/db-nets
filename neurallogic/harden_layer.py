import jax

from neurallogic import neural_logic_net


def logistic_clip(x):
    return jax.scipy.special.expit(3 * (2 * x - 1))


def harden(x):
    # non-differentiable
    return jax.lax.cond(x > 0.5, lambda _: 1.0, lambda _: 0.0, None)


def scaled_straight_through_harden(x):
    # The harden operation is non-differentiable. Therefore we need to
    # approximate with the straight-through estimator.

    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = x - jax.lax.stop_gradient(x)
    one = zero + jax.lax.stop_gradient(harden(x))

    # However, the straight-through estimator discards information about the value of x.
    # In consequence, backprogated errors are independent of whether x is close to
    # the decision boundary at 0.5. This is undesirable because we only want to 
    # allocate soft weight resources to the region around the decision boundary.
    # We therefore scale the gradient to be smaller at x=0.5 and unscaled at 
    # x=0.0 and x=1.0. In other words, we minimally update upstream weights in order
    # to potentially flip the hard value of x.
    scale_factor = (2 * (0.5 - x)) * (2 * (0.5 - x)) + 0.001
    return one * scale_factor


soft_harden_layer = jax.vmap(scaled_straight_through_harden)


def hard_harden_layer(x):
    return x


# TODO: can we harden arbitrary tensors?
# TODO: is this correct?
def symbolic_harden_layer(x):
    return x


harden_layer = neural_logic_net.select(
    soft_harden_layer, hard_harden_layer, symbolic_harden_layer
)
 