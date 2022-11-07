import jax.numpy as jnp
from neurallogic import hard_not
from neurallogic import harden

def test_activation():
    test_data = [[[1.0, 1.0], 1.0], [[1.0, 0.0], 0.0], [[0.0, 0.0], 1.0], [[0.0, 1.0], 0.0]]
    for input, expected in test_data:
        assert hard_not.soft_not(*input) == expected
        assert hard_not.hard_not(*harden.harden_list(input)) == harden.harden(expected)

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
        assert jnp.array_equal(hard_not.hard_not_neuron(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input))), harden.harden_array(jnp.array(expected)))

def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        [[1.0, 1.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]],
        [[0.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    ]
    for input, weights, expected in test_data:
        assert jnp.array_equal(hard_not.soft_not_layer(jnp.array(weights), jnp.array(input)), jnp.array(expected))
        assert jnp.array_equal(hard_not.hard_not_layer(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input))), harden.harden_array(jnp.array(expected)))
