import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random

from neurallogic import hard_not, harden, neural_logic_net, primitives


def test_activation():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 0.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 0.0]
    ]
    for input, expected in test_data:
        assert hard_not.soft_not(*input) == expected
        assert hard_not.hard_not(*harden.harden(input)) == harden.harden(expected)
        assert eval(str(hard_not.symbolic_not(*harden.harden(input)))) == harden.harden(expected)

def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    ]
    for input, weights, expected in test_data:
        assert jnp.array_equal(hard_not.soft_not_neuron(jnp.array(weights), jnp.array(input)), jnp.array(expected))
        assert jnp.array_equal(hard_not.hard_not_neuron(harden.harden(jnp.array(weights)), harden.harden(jnp.array(input))), harden.harden(jnp.array(expected)))
        assert jnp.array_equal(eval(str(hard_not.symbolic_not_neuron(harden.harden(jnp.array(weights)).tolist(), harden.harden(jnp.array(input)).tolist()))), harden.harden(jnp.array(expected)))

def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]], [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.8]]],
        [[1.0, 0.4], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 0.4], [0.0, 0.4], [1.0, 0.6], [0.0, 0.6]]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.8], [0.0, 0.0], [1.0, 0.0]]],
        [[0.0, 0.0], [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 0.99], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    ]
    for input, weights, expected in test_data:
        assert jnp.array_equal(hard_not.soft_not_layer(jnp.array(weights), jnp.array(input)), jnp.array(expected))
        assert jnp.array_equal(hard_not.hard_not_layer(harden.harden(jnp.array(weights)), harden.harden(jnp.array(input))), harden.harden(jnp.array(expected)))

def test_not():
    def test_net(type, x):
        x = hard_not.NotLayer(4, type)(x)
        x = primitives.ravel(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    test_data = [
        [
            [1.0, 1.0],
            [0.9469013, 0.679816, 0.3194083, 0.41585994, 0.7815013, 0.9580679, 0.2925768, 0.02594423]
        ],
        [
            [1.0, 0.0],
            [0.9469013, 0.320184, 0.3194083, 0.58414006, 0.7815013, 0.04193211, 0.2925768, 0.97405577]
        ],
        [
            [0.0, 1.0],
            [0.05309868, 0.679816, 0.6805917, 0.41585994, 0.2184987, 0.9580679, 0.7074232, 0.02594423]
        ],
        [
            [0.0, 0.0],
            [0.05309868, 0.320184, 0.6805917, 0.58414006, 0.2184987, 0.04193211, 0.7074232, 0.97405577]
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

def test_train_not():
    def test_net(type, x):
        return hard_not.NotLayer(4, type)(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ]
    input = jnp.array(x)
    output = jnp.array(y)

    # Train the not layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(apply_fn=jax.vmap(soft.apply, in_axes=(None, 0)), params=soft_weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x, y: jnp.mean((state.apply_fn(params, x) - y) ** 2)))
    for epoch in range(1, 100):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the not layer (both soft and hard variants) correctly predicts y
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

def test_symbolic_not():
    def test_net(type, x):
        x = hard_not.NotLayer(4, type)(x)
        x = primitives.ravel(type)(x)
        x = hard_not.NotLayer(4, type)(x)
        x = primitives.ravel(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    symbolic_weights = harden.symbolic_weights(soft_weights)
    symbolic_input = ['x1', 'x2']
    symbolic_result = symbolic.apply(symbolic_weights, symbolic_input)
    assert(symbolic_result == ['(not((not(x1 ^ True)) ^ True))', '(not((not(x2 ^ True)) ^ True))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ False))', '(not((not(x1 ^ True)) ^ False))', '(not((not(x2 ^ True)) ^ True))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ False))', '(not((not(x1 ^ True)) ^ True))', '(not((not(x2 ^ True)) ^ False))', '(not((not(x1 ^ False)) ^ True))', '(not((not(x2 ^ False)) ^ True))', '(not((not(x1 ^ True)) ^ False))', '(not((not(x2 ^ True)) ^ False))', '(not((not(x1 ^ False)) ^ True))', '(not((not(x2 ^ False)) ^ True))', '(not((not(x1 ^ True)) ^ False))', '(not((not(x2 ^ True)) ^ True))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ False))', '(not((not(x1 ^ True)) ^ True))', '(not((not(x2 ^ True)) ^ True))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ True))', '(not((not(x1 ^ True)) ^ True))', '(not((not(x2 ^ True)) ^ False))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ False))', '(not((not(x1 ^ True)) ^ False))', '(not((not(x2 ^ True)) ^ True))', '(not((not(x1 ^ False)) ^ False))', '(not((not(x2 ^ False)) ^ False))'])

