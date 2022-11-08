import jax.numpy as jnp
from jax import random
from neurallogic import hard_not
from neurallogic import harden

def test_activation():
    test_data = [
        # Test logic
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 0.0],
        # test clipping
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 0.0]
    ]
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

def test_soft_not():
    softNot = hard_not.SoftNOTLayer(layer_size=4)
    weights = softNot.init(random.PRNGKey(0), [1.0, 1.0])
    test_data = [
        [
            [1.0, 1.0],
            [[0.26377404, 0.8707025], [0.44444847, 0.03216302], [0.6110164, 0.685097], [0.9133855, 0.08662593]]
        ],
        [
            [1.0, 0.0],
            [[0.26377404, 0.1292975], [0.44444847, 0.967837], [0.6110164, 0.31490302], [0.9133855, 0.91337407]]
        ],
        [
            [0.0, 1.0],
            [[0.73622596, 0.8707025 ], [0.5555515, 0.03216302], [0.3889836, 0.685097], [0.08661449, 0.08662593]]
        ],
        [
            [0.0, 0.0],
            [[0.73622596, 0.1292975], [0.5555515, 0.967837], [0.3889836, 0.31490302], [0.08661449, 0.91337407]]
        ]
    ]
    for input, expected in test_data:
        input = jnp.array(input)
        expected = jnp.array(expected)
        result = softNot.apply(weights, input)
        assert jnp.allclose(result, expected)
