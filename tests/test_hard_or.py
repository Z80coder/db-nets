import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

from neurallogic import hard_or, harden, neural_logic_net, primitives


def test_include():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 0.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 0.0],
        [[-0.1, 1.0], 0.0]
    ]
    for input, expected in test_data:
        assert hard_or.soft_or_include(*input) == expected
        assert hard_or.hard_or_include(*harden.harden(input)) == harden.harden(expected)
        symbolic_output = hard_or.symbolic_or_include(*harden.harden(input))
        assert symbolic_output == harden.harden(expected)

def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], 1.0],
        [[0.0, 0.0], [0.0, 0.0], 0.0],
        [[1.0, 0.0], [0.0, 1.0], 0.0],
        [[0.0, 1.0], [1.0, 0.0], 0.0],
        [[0.0, 1.0], [0.0, 0.0], 0.0],
        [[0.0, 1.0], [1.0, 1.0], 1.0]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        assert jnp.allclose(hard_or.soft_or_neuron(weights, input), expected)
        assert jnp.allclose(hard_or.hard_or_neuron(harden.harden(weights), harden.harden(input)), harden.harden(expected))
        symbolic_output = hard_or.symbolic_or_neuron(harden.harden(weights.tolist()), harden.harden(input.tolist()))
        assert jnp.array_equal(symbolic_output, harden.harden(expected))

def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]], [1.0, 0.0, 1.0, 0.0]],
        [[1.0, 0.4], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [1.0, 0.39999998, 1.0, 0.0]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]], [1.0, 0.8, 0.0, 0.0]],
        [[0.0, 0.0], [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0, 0.0]]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        expected = jnp.array(expected)
        assert jnp.allclose(hard_or.soft_or_layer(weights, input), expected)
        assert jnp.allclose(hard_or.hard_or_layer(harden.harden(weights), harden.harden(input)), harden.harden(expected))
        symbolic_output = hard_or.symbolic_or_layer(harden.harden(weights.tolist()), harden.harden(input.tolist()))
        assert jnp.array_equal(symbolic_output, harden.harden(expected))

def test_or():
    def test_net(type, x):
        x = hard_or.or_layer(4, type, nn.initializers.uniform(1.0))(x)
        x = primitives.nl_ravel(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    test_data = [
        [
            [1.0, 1.0],
            [0.45491087, 0.36511207, 0.62628365, 0.95989954]
        ],
        [
            [1.0, 0.0],
            [0.2715416, 0.03128195, 0.01773429, 0.5896025]
        ],
        [
            [0.0, 1.0],
            [0.45491087, 0.36511207, 0.62628365, 0.95989954]
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
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
        assert jnp.allclose(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)

def test_train_or():
    def test_net(type, x):
        return hard_or.or_layer(4, type, nn.initializers.uniform(1.0))(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
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
        assert jnp.allclose(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)

def test_symbolic_or():
    def test_net(type, x):
        x = hard_or.or_layer(4, type, nn.initializers.uniform(1.0))(x)
        x = hard_or.or_layer(4, type, nn.initializers.uniform(1.0))(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    symbolic_weights = harden.symbolic_weights(soft_weights)
    symbolic_input = ['x1', 'x2']
    symbolic_result = symbolic.apply(symbolic_weights, symbolic_input)
    assert(symbolic_result == ['((((x1 and False) or (x2 and False)) and True) or (((x1 and False) or (x2 and False)) and True) or (((x1 and False) or (x2 and True)) and True) or (((x1 and True) or (x2 and True)) and True))', '((((x1 and False) or (x2 and False)) and False) or (((x1 and False) or (x2 and False)) and True) or (((x1 and False) or (x2 and True)) and False) or (((x1 and True) or (x2 and True)) and False))', '((((x1 and False) or (x2 and False)) and False) or (((x1 and False) or (x2 and False)) and False) or (((x1 and False) or (x2 and True)) and True) or (((x1 and True) or (x2 and True)) and True))', '((((x1 and False) or (x2 and False)) and False) or (((x1 and False) or (x2 and False)) and True) or (((x1 and False) or (x2 and True)) and True) or (((x1 and True) or (x2 and True)) and True))'])

