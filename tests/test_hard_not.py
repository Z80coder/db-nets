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

def test_not():
    soft_net, hard_net = hard_not.NotLayer(layer_size=4)
    soft_weights = soft_net.init(random.PRNGKey(0), [1.0, 1.0])
    hard_weights = harden.harden_dict(soft_weights)
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
        soft_input = jnp.array(input)
        hard_input = harden.harden_array(soft_input)
        soft_expected = jnp.array(expected)
        hard_expected = harden.harden_array(soft_expected)
        soft_result = soft_net.apply(soft_weights, soft_input)
        assert jnp.allclose(soft_result, soft_expected)
        hard_result = hard_net.apply(hard_weights, hard_input)
        assert jnp.array_equal(hard_result, hard_expected)



