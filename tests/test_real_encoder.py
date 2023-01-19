from typing import Callable

import jax
from jax.config import config
import numpy
import optax
from flax.training import train_state
from jax import random

from neurallogic import harden, neural_logic_net, real_encoder, symbolic_generation


config.update("jax_debug_nans", True)


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

    test_data = [
        [
            [1.0, 0.8],
            [[1.0, 1.0, 1.0], [0.47898874, 0.4623352, 0.6924789]],
        ],
        [
            [0.6, 0.0],
            [
                [0.78442293, 0.7857669, 0.3154459],
                [0.0, 0.0, 0.0],
            ],
        ],
        [
            [0.1, 0.9],
            [
                [0.5149515, 0.51797545, 0.05257431],
                [0.69679934, 0.629154, 0.84623945],
            ],
        ],
        [
            [0.4, 0.6],
            [
                [0.6766343, 0.67865026, 0.21029726],
                [0.35924158, 0.34675142, 0.4445637],
            ],
        ],
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        soft_output = soft.apply(weights, jax.numpy.array(input))
        soft_expected = jax.numpy.array(expected)
        assert jax.numpy.allclose(soft_output, soft_expected)

        # Check that the hard function performs as expected
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, jax.numpy.array(input))
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, jax.numpy.array(input))
        assert numpy.allclose(symbolic_output, hard_expected)


def test_train_real_encoder():
    def test_net(type, x):
        return real_encoder.real_encoder_layer(type)(3)(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])

    x = [
        [0.8, 0.9],
        [0.85, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
    ]
    y = [
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
    ]
    input = jax.numpy.array(x)
    output = jax.numpy.array(y)

    # Train the real_encoder layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(
        apply_fn=jax.vmap(soft.apply, in_axes=(None, 0)), params=weights, tx=tx
    )
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda params, x, y: jax.numpy.mean((state.apply_fn(params, x) - y) ** 2)
        )
    )
    for epoch in range(1, 100):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the real_encoder layer (both soft and hard variants) correctly predicts y
    weights = state.params
    hard_weights = harden.hard_weights(weights)

    for input, expected in zip(x, y):
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_result = hard.apply(hard_weights, jax.numpy.array(input))
        assert jax.numpy.allclose(hard_result, hard_expected)
        symbolic_output = symbolic.apply(hard_weights, jax.numpy.array(input))
        assert jax.numpy.array_equal(symbolic_output, hard_expected)
