from typing import Callable
import numpy
import jax
from jax import random

from neurallogic import harden, real_encoder, symbolic_generation, neural_logic_net


def check_consistency(soft: Callable, hard: Callable, expected, *args):
    # print(f'\nchecking consistency for {soft.__name__}')
    # Check that the soft function performs as expected
    soft_output = soft(*args)
    # print(f'Expected: {expected}, Actual soft_output: {soft_output}')
    assert numpy.allclose(soft_output, expected, equal_nan=True)

    # Check that the hard function performs as expected
    # N.B. We don't harden the inputs because the hard_bit expects real-valued inputs
    hard_expected = harden.harden(expected)
    hard_output = hard(*args)
    # print(f'Expected: {hard_expected}, Actual hard_output: {hard_output}')
    assert numpy.allclose(hard_output, hard_expected, equal_nan=True)

    # Check that the jaxpr performs as expected
    symbolic_f = symbolic_generation.make_symbolic_jaxpr(hard, *args)
    symbolic_output = symbolic_generation.eval_symbolic(symbolic_f, *args)
    # print(f'Expected: {hard_expected}, Actual symbolic_output: {symbolic_output}')
    assert numpy.allclose(symbolic_output, hard_expected, equal_nan=True)


def test_activation():
    test_data = [
        [[1.0, 1.0], 0.5],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 0.5],
        [[0.0, 1.0], 1.0],
        [[0.6, 1.0], 1.0],
        [[0.8, 0.75], 0.46875],
        [[0.3, 0.1], 0.1666666716337204],
        [[0.25, 0.9], 0.933333337306976],
    ]
    for input, expected in test_data:
        check_consistency(
            real_encoder.soft_real_encoder,
            real_encoder.hard_real_encoder,
            expected,
            input[0],
            input[1],
        )


def test_neuron():
    test_data = [
        [1.0, [1.0, 1.0, 0.6], [0.5, 0.5, 0.99999994]],
        [0.0, [0.0, 0.0, 0.9], [0.5, 0.5, 0.0]],
        [1.0, [0.0, 1.0, 0.1], [1.0, 0.5, 1.0]],
        [0.0, [1.0, 0.0, 0.3], [0.0, 0.5, 0.0]],
        [0.3, [0.2, 0.8, 0.3], [0.5625, 0.1875, 0.5]],
        [0.1, [0.9, 0.42, 0.5], [0.05555556, 0.11904762, 0.1]],
        [0.4, [0.2, 0.8, 0.7], [0.625, 0.25, 0.2857143]],
        [0.32, [0.9, 0.1, 0.01], [0.17777778, 0.6222222, 0.6565656]],
    ]
    for input, thresholds, expected in test_data:

        def soft(thresholds, input):
            return real_encoder.soft_real_encoder_neuron(thresholds, input)

        def hard(thresholds, input):
            return real_encoder.hard_real_encoder_neuron(thresholds, input)

        check_consistency(
            soft, hard, expected, jax.numpy.array(thresholds), jax.numpy.array(input)
        )


def test_layer():
    test_data = [
        [
            [1.0, 0.0],
            [[1.0, 1.0, 0.3], [0.0, 1.0, 0.4]],
            [[0.5, 0.5, 1.0], [0.5, 0.0, 0.0]],
        ],
        [
            [1.0, 0.4],
            [[1.0, 1.0, 0.1], [0.0, 1.0, 0.9]],
            [[0.5, 0.5, 1.0], [0.7, 0.2, 0.22222224]],
        ],
        [
            [0.0, 1.0],
            [[1.0, 1.0, 0.2], [0.0, 0.8, 0.75]],
            [[0.0, 0.0, 0.0], [1.0, 0.99999994, 1.0]],
        ],
        [
            [0.45, 0.95],
            [[1.0, 0.01, 0.94], [0.0, 1.0, 0.49]],
            [[0.225, 0.7222222, 0.23936169], [0.975, 0.475, 0.9509804]],
        ],
    ]
    for input, thresholds, expected in test_data:

        def soft(thresholds, input):
            return real_encoder.soft_real_encoder_layer(thresholds, input)

        def hard(thresholds, input):
            return real_encoder.hard_real_encoder_layer(thresholds, input)

        check_consistency(
            soft,
            hard,
            jax.numpy.array(expected),
            jax.numpy.array(thresholds),
            jax.numpy.array(input),
        )


def test_real_encoder():
    def test_net(type, x):
        return real_encoder.real_encoder_layer(type)(3)(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)
    print(f'weights: {weights}')
    print(f'hard_weights: {hard_weights}')

    test_data = [
        [
            [1.0, 0.8],
            [
                [1.0, 1.0, 1.0],
                [0.47898874, 0.4623352,  0.6924789]
            ],
        ],
        [
            [0.6, 0.0],
            [
                [
                    0.9469013,
                    0.320184,
                    0.3194083,
                ],
                [
                    0.58414006,
                    0.7815013,
                    0.04193211,
                ],
            ],
        ],
        [
            [0.1, 0.9],
            [
                [
                    0.05309868,
                    0.679816,
                    0.6805917,
                ],
                [
                    0.41585994,
                    0.2184987,
                    0.9580679,
                ],
            ],
        ],
        [
            [0.4, 0.6],
            [
                [
                    0.05309868,
                    0.320184,
                    0.6805917,
                ],
                [
                    0.58414006,
                    0.2184987,
                    0.04193211,
                ],
            ],
        ],
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        soft_output = soft.apply(weights, jax.numpy.array(input))
        soft_expected = jax.numpy.array(expected)
        print(f'soft_output: {soft_output}\nsoft_expected: {soft_expected}')
        assert jax.numpy.allclose(soft_output, soft_expected)

        # Check that the hard function performs as expected
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, jax.numpy.array(input))
        print(f'hard_output: {hard_output}\nhard_expected: {hard_expected}')
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, jax.numpy.array(input))
        print(f'symbolic_output: {symbolic_output}\nhard_expected: {hard_expected}')
        assert numpy.allclose(symbolic_output, hard_expected)
