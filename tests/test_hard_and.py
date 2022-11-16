import jax
import jax.numpy as jnp
from jax import random

from neurallogic import hard_and, harden, neural_logic_net, primitives, hard_not

import optax
from flax.training import train_state

def test_include():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 1.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 1.0]
    ]
    for input, expected in test_data:
        assert hard_and.soft_and_include(*input) == expected
        assert hard_and.hard_and_include(*harden.harden(input)) == harden.harden(expected)

def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], 1.0],
        [[0.0, 0.0], [0.0, 0.0], 1.0],
        [[1.0, 0.0], [0.0, 1.0], 0.0],
        [[0.0, 1.0], [1.0, 0.0], 0.0],
        [[0.0, 1.0], [0.0, 0.0], 1.0],
        [[0.0, 1.0], [1.0, 1.0], 0.0]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        assert jnp.array_equal(hard_and.soft_and_neuron(weights, input), expected)
        assert jnp.array_equal(hard_and.hard_and_neuron(harden.harden(weights), harden.harden(input)), harden.harden(expected))

def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]], [0.0, 0.0, 1.0, 0.8]],
        [[1.0, 0.4], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [0.4, 0.4, 1.0, 1.0]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]], [0.0, 1.0, 0.0, 1.0]],
        [[0.0, 0.0], [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0, 1.0]]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        expected = jnp.array(expected)
        assert jnp.array_equal(hard_and.soft_and_layer(weights, input), expected)
        assert jnp.array_equal(hard_and.hard_and_layer(harden.harden(weights), harden.harden(input)), harden.harden(expected))

def test_and():
    def test_net(type, x):
        x = hard_and.AndLayer(4, type)(x)
        x = primitives.ravel(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    test_data = [
        [
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
        [
            [1.0, 0.0],
            [0.53707814, 0.13892889, 0.7197356, 0.62835324]
        ],
        [
            [0.0, 1.0],
            [0.35097015, 0.34810758, 0.04889798, 0.43687034]
        ],
        [
            [0.0, 0.0],
            [0.35097015, 0.13892889, 0.04889798, 0.43687034]
        ]
    ]
    for input, expected in test_data:
        soft_input = jnp.array(input)
        soft_expected = jnp.array(expected)
        soft_result = soft.apply(soft_weights, soft_input)
        assert jnp.allclose(soft_result, soft_expected)
        hard_input = harden.harden(soft_input)
        hard_expected = harden.harden(soft_expected)
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.array_equal(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)

def test_train_and():
    def test_net(type, x):
        return hard_and.AndLayer(4, type)(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0]
    ]
    input = jnp.array(x)
    output = jnp.array(y)

    # Train the and layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(apply_fn=jax.vmap(soft.apply, in_axes=(None, 0)), params=soft_weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x, y: jnp.mean((state.apply_fn(params, x) - y) ** 2)))
    for epoch in range(1, 100):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the and layer (both soft and hard variants) correctly predicts y
    soft_weights = state.params
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    for input, expected in zip(x, y):
        hard_input = harden.harden_array(harden.harden(jnp.array(input)))
        hard_expected = harden.harden_array(harden.harden(jnp.array(expected)))
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.array_equal(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)

def test_symbolic_and():
    def test_net(type, x):
        x = hard_and.AndLayer(4, type)(x)
        x = hard_and.AndLayer(4, type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    symbolic_weights = harden.symbolic_weights(soft_weights)
    symbolic_input = ['x1', 'x2']
    symbolic_result = symbolic.apply(symbolic_weights, symbolic_input)
    assert(symbolic_result == ['((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))', '((((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))', '((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(False)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))', '((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)))'])

