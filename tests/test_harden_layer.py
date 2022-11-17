import jax.numpy as jnp

from neurallogic import harden_layer, harden

def test_harden_layer():
    test_data = [
        [[0.8, 0.1], [1.0, 0.0]],
        [[1.0, 0.52], [1.0, 1.0]],
        [[0.3, 0.51], [0.0, 1.0]],
        [[0.49, 0.32], [0.0, 0.0]]
    ]
    for input, expected in test_data:
        input = jnp.array(input)
        expected = jnp.array(expected)
        assert jnp.array_equal(harden_layer.soft_harden_layer(input), expected)
        assert jnp.array_equal(harden_layer.hard_harden_layer(harden.harden(input)), harden.harden(expected))
        symbolic_output = harden_layer.symbolic_harden_layer(harden.harden(input.tolist()))
        assert jnp.array_equal(symbolic_output, harden.harden(expected))
