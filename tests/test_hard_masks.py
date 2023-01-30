import jax
import numpy
from jax import random

from neurallogic import hard_masks, harden, neural_logic_net
from tests import utils


def test_mask_to_true():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 1.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 1.0],
    ]
    for input, expected in test_data:
        utils.check_consistency(
            hard_masks.soft_mask_to_true,
            hard_masks.hard_mask_to_true,
            expected,
            input[0],
            input[1],
        )


def test_mask_to_true_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_masks.soft_mask_to_true_neuron(weights, input)

        def hard(weights, input):
            return hard_masks.hard_mask_to_true_neuron(weights, input)

        utils.check_consistency(
            soft, hard, expected, jax.numpy.array(weights), jax.numpy.array(input)
        )


def test_mask_to_true_layer():
    test_data = [
        [
            [1.0, 0.0],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.8]],
        ],
        [
            [1.0, 0.4],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.4], [1.0, 0.4], [1.0, 1.0], [1.0, 1.0]],
        ],
        [
            [0.0, 1.0],
            [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        ],
        [
            [0.0, 0.0],
            [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.99], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_masks.soft_mask_to_true_layer(weights, input)

        def hard(weights, input):
            return hard_masks.hard_mask_to_true_layer(weights, input)

        utils.check_consistency(
            soft,
            hard,
            jax.numpy.array(expected),
            jax.numpy.array(weights),
            jax.numpy.array(input),
        )


def test_mask_to_true():
    def test_net(type, x):
        x = hard_masks.mask_to_true_layer(type)(4)(x)
        x = x.ravel()
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)

    test_data = [
        [
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        [
            [1.0, 0.0],
            [1.0, 0.17739451, 1.0, 0.77752244, 1.0, 0.11280203, 1.0, 0.43465567],
        ],
        [
            [0.0, 1.0],
            [0.6201445, 1.0, 0.7178699, 1.0, 0.29197645, 1.0, 0.41213453, 1.0],
        ],
        [
            [0.0, 0.0],
            [
                0.6201445,
                0.17739451,
                0.7178699,
                0.77752244,
                0.29197645,
                0.11280203,
                0.41213453,
                0.43465567,
            ],
        ],
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        soft_output = soft.apply(weights, jax.numpy.array(input))
        expected_output = jax.numpy.array(expected)
        assert jax.numpy.allclose(soft_output, expected_output)

        # Check that the hard function performs as expected
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, hard_input)
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert numpy.allclose(symbolic_output, hard_expected)


def test_mask_to_false():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 0.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 0.0],
        [[-0.1, 1.0], 0.0],
    ]
    for input, expected in test_data:
        utils.check_consistency(
            hard_masks.soft_mask_to_false,
            hard_masks.hard_mask_to_false,
            expected,
            input[0],
            input[1],
        )


def test_mask_to_false_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_masks.soft_mask_to_false_neuron(weights, input)

        def hard(weights, input):
            return hard_masks.hard_mask_to_false_neuron(weights, input)

        utils.check_consistency(
            soft, hard, expected, jax.numpy.array(weights), jax.numpy.array(input)
        )


def test_mask_to_false_layer():
    test_data = [
        [
            [1.0, 0.0],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]],
            [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        ],
        [
            [1.0, 0.4],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.39999998], [0.0, 0.39999998], [1.0, 0.0], [0.0, 0.0]],
        ],
        [
            [0.0, 1.0],
            [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [0.0, 0.0],
            [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_masks.soft_mask_to_false_layer(weights, input)

        def hard(weights, input):
            return hard_masks.hard_mask_to_false_layer(weights, input)

        utils.check_consistency(
            soft,
            hard,
            jax.numpy.array(expected),
            jax.numpy.array(weights),
            jax.numpy.array(input),
        )


def test_mask_to_false():
    def test_net(type, x):
        x = hard_masks.mask_to_false_layer(type)(4)(x)
        x = x.ravel()
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)

    test_data = [
        [
            [1.0, 1.0],
            [
                0.3798555,
                0.8226055,
                0.28213012,
                0.22247756,
                0.70802355,
                0.887198,
                0.5878655,
                0.56534433,
            ],
        ],
        [
            [1.0, 0.0],
            [0.3798555, 0.0, 0.28213012, 0.0, 0.70802355, 0.0, 0.5878655, 0.0],
        ],
        [
            [0.0, 1.0],
            [0.0, 0.8226055, 0.0, 0.22247756, 0.0, 0.887198, 0.0, 0.56534433]
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        soft_output = soft.apply(weights, jax.numpy.array(input))
        expected_output = jax.numpy.array(expected)
        assert jax.numpy.allclose(soft_output, expected_output)

        # Check that the hard function performs as expected
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, hard_input)
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert numpy.allclose(symbolic_output, hard_expected)
