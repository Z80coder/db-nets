from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

from neurallogic import (hard_and, hard_not, hard_or, harden, harden_layer,
                         neural_logic_net, primitives)


def test_train_network():
    def test_net(type, x):
        x = hard_or.or_layer(type)(16, nn.initializers.uniform(1.0), jnp.float64)(x)
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0), jnp.float64)(x)
        x = hard_not.not_layer(type)(1, dtype=jnp.float64)(x)
        x = primitives.nl_ravel(type)(x)
        x = harden_layer.harden_layer(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
    y = [
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0]
    ]
    input = jnp.array(x)
    output = jnp.array(y)

    # Train the and layer
    tx = optax.sgd(0.01)
    state = train_state.TrainState.create(apply_fn=jax.vmap(soft.apply, in_axes=(None, 0)), params=soft_weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x, y: jnp.mean((state.apply_fn(params, x) - y) ** 2)))
    min_loss = 1e10
    best_weights = None
    for epoch in range(1, 500):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)
        if loss < min_loss:
            min_loss = loss
            best_weights = state.params

    # Test that the and layer (both soft and hard variants) correctly predicts y
    for input, expected in zip(x, y):
        input = jnp.array(input)
        expected = jnp.array(expected)
        soft_result = soft.apply(best_weights, input)
        assert jnp.allclose(soft_result, expected)
        hard_input = harden.harden(input)
        hard_expected = harden.harden(expected)
        hard_weights = harden.hard_weights(best_weights)
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.array_equal(hard_result, hard_expected)
        symbolic_weights = harden.symbolic_weights(best_weights)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)
        symbolic_input = ['x1', 'x2']
        symbolic_expected = ['(not(((((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and False) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and False)) or not(False)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True))) ^ False))', '(not(((((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and False)) or not(False)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False))) ^ True))', '(not(((((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and False) or (x2 and False)) or not(True)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True))) ^ False))', '(not(((((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and False)) or not(False)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and False)) or not(True)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and True) or (x2 and True)) or not(False)) and (((x1 and False) or (x2 and False)) or not(True)) and (((x1 and False) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True)) and (((x1 and True) or (x2 and True)) or not(True))) ^ True))']
        symbolic_result = symbolic.apply(symbolic_weights, symbolic_input)
        assert symbolic_result == symbolic_expected
