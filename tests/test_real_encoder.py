from typing import Callable

import jax
import numpy
import optax
from flax.training import train_state
from jax import random
from jax.config import config

from neurallogic import (harden, neural_logic_net, real_encoder,
                         symbolic_generation)
from tests import utils

# Uncomment to debug NaNs
# config.update("jax_debug_nans", True)


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
            soft, hard, expected, jax.numpy.array(
                thresholds), jax.numpy.array(input)
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
            lambda params, x, y: jax.numpy.mean(
                (state.apply_fn(params, x) - y) ** 2)
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


def test_symbolic_real_encoder():
    def test_net(type, x):
        x = real_encoder.real_encoder_layer(type)(3)(x)
        # x = x.ravel()
        # x = real_encoder.real_encoder_layer(type)(4)(x)
        # x = x.ravel()
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)

    # Compute soft result
    soft_input = jax.numpy.array([1.0, 0.0])
    weights = soft.init(random.PRNGKey(0), soft_input)
    soft_result = soft.apply(weights, numpy.array(soft_input))

    # Compute hard result
    hard_weights = harden.hard_weights(weights)
    hard_result = hard.apply(hard_weights, numpy.array(soft_input))
    # Check that the hard result is the same as the soft result
    assert numpy.array_equal(harden.harden(soft_result), hard_result)

    # Compute symbolic result with non-symbolic inputs
    symbolic_output = symbolic.apply(hard_weights, soft_input)
    # Check that the symbolic result is the same as the hard result
    assert numpy.array_equal(symbolic_output, hard_result)

    # Compute symbolic result with symbolic inputs and symbolic weights, but where the symbols can be evaluated
    symbolic_input = ['1.0', '0.0']
    symbolic_weights = utils.make_symbolic(hard_weights)
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_input)
    symbolic_output = symbolic_generation.eval_symbolic_expression(
        symbolic_output)
    # Check that the symbolic result is the same as the hard result
    assert numpy.array_equal(symbolic_output, hard_result)

    # Compute symbolic result with symbolic inputs and non-symbolic weights
    symbolic_input = ['x1', 'x2']
    symbolic_output = symbolic.apply(hard_weights, symbolic_input)
    # Check the form of the symbolic expression
    assert numpy.array_equal(
        symbolic_output,
        r'lax_reference.select(numpy.array(lax_reference.gt(lax_reference.select(numpy.numpy.array([[numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.07225775718688965, x1)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(0.07225775718688965, x1))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x1, x1)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.06643760204315186, x1)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(0.06643760204315186, x1))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x1, x1)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.9510347843170166, x1)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(0.9510347843170166, x1))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x1, x1))))],       [numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.8350926637649536, x2)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(0.8350926637649536, x2))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x2, x2)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.8651731014251709, x2)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(0.8651731014251709, x2))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x2, x2)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(0.6748189926147461, x2)), numpy.add(9.99999993922529e-09, numpy.multiply(9.999999747378752e-06, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(False, lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(False, lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(0.6748189926147461, x2))), numpy.logical_not(numpy.logical_or(False, lax_reference.ne(x2, x2))))]],      dtype=object), numpy.numpy.array([[0.5, 0.5, 0.5],       [0.5, 0.5, 0.5]], dtype=numpy.numpy.float32), lax_reference.select(numpy.numpy.array([[lax_reference.lt(x1, 0.07225775718688965),        lax_reference.lt(x1, 0.06643760204315186),        lax_reference.lt(x1, 0.9510347843170166)],       [lax_reference.lt(x2, 0.8350926637649536),        lax_reference.lt(x2, 0.8651731014251709),        lax_reference.lt(x2, 0.6748189926147461)]], dtype=object), numpy.numpy.array([[numpy.multiply(6.9196672439575195, x1),        numpy.multiply(7.525852680206299, x1),        numpy.multiply(0.5257430672645569, x1)],       [numpy.multiply(0.5987359285354614, x2),        numpy.multiply(0.5779190063476562, x2),        numpy.multiply(0.7409393787384033, x2)]], dtype=object), numpy.numpy.array([[numpy.multiply(0.5389427542686462, numpy.subtract(numpy.add(x1, 1.0), 0.1445155143737793)),        numpy.multiply(0.5355827808380127, numpy.subtract(numpy.add(x1, 1.0), 0.1328752040863037)),        numpy.multiply(10.211319923400879, numpy.subtract(numpy.add(x1, 1.0), 1.9020695686340332))],       [numpy.multiply(3.0320050716400146, numpy.subtract(numpy.add(x2, 1.0), 1.6701853275299072)),        numpy.multiply(3.7084574699401855, numpy.subtract(numpy.add(x2, 1.0), 1.7303462028503418)),        numpy.multiply(1.5376049280166626, numpy.subtract(numpy.add(x2, 1.0), 1.3496379852294922))]],      dtype=object))), 0.5),      dtype=object), numpy.array([[ True,  True,  True],       [ True,  True,  True]]), numpy.array([[False, False, False],       [False, False, False]]))'
    )

    # Compute symbolic result with symbolic inputs and symbolic weights
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_input)
    # Check the form of the symbolic expression
    # N.B. expected output can change depending due to presence of small numerical errors that can differ between runs and platforms
    expected_output = r'lax_reference.select(lax_reference.gt(lax_reference.select(numpy.array([[numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776)), x1)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))), inf), lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))), inf), lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776)), x1))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776)), numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))), lax_reference.ne(x1, x1)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376)), x1)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))), inf), lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))), inf), lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376)), x1))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376)), numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))), lax_reference.ne(x1, x1)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)), x1)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x1)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348))), inf), lax_reference.eq(numpy.absolute(x1), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348))), inf), lax_reference.eq(numpy.absolute(x1), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)), x1))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)), numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348))), lax_reference.ne(x1, x1))))],       [numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266)), x2)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))), inf), lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))), inf), lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266)), x2))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266)), numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))), lax_reference.ne(x2, x2)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731)), x2)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))), inf), lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))), inf), lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731)), x2))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731)), numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))), lax_reference.ne(x2, x2)))),        numpy.logical_and(numpy.logical_or(numpy.logical_and(lax_reference.le(numpy.absolute(numpy.subtract(numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)), x2)), numpy.add(1e-08, numpy.multiply(1e-05, numpy.absolute(x2)))), numpy.logical_not(numpy.logical_or(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.674819))), inf), lax_reference.eq(numpy.absolute(x2), inf)))), numpy.logical_and(numpy.logical_and(lax_reference.eq(numpy.absolute(numpy.minimum(1.0, numpy.maximum(0.0, 0.674819))), inf), lax_reference.eq(numpy.absolute(x2), inf)), lax_reference.eq(numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)), x2))), numpy.logical_not(numpy.logical_or(lax_reference.ne(numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)), numpy.minimum(1.0, numpy.maximum(0.0, 0.674819))), lax_reference.ne(x2, x2))))]],      dtype=object), numpy.array([[0.5, 0.5, 0.5],       [0.5, 0.5, 0.5]]), lax_reference.select(numpy.array([[lax_reference.lt(x1, numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))),        lax_reference.lt(x1, numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))),        lax_reference.lt(x1, numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)))],       [lax_reference.lt(x2, numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))),        lax_reference.lt(x2, numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))),        lax_reference.lt(x2, numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)))]],      dtype=object), numpy.array([[numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))), 1e-07)), x1),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))), 1e-07)), x1),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348))), 1e-07)), x1)],       [numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))), 1e-07)), x2),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))), 1e-07)), x2),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.674819))), 1e-07)), x2)]],      dtype=object), numpy.array([[numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776)))), 1e-07)), numpy.subtract(numpy.add(x1, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.07225776))))),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376)))), 1e-07)), numpy.subtract(numpy.add(x1, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.0664376))))),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)))), 1e-07)), numpy.subtract(numpy.add(x1, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.9510348)))))],       [numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266)))), 1e-07)), numpy.subtract(numpy.add(x2, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.83509266))))),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731)))), 1e-07)), numpy.subtract(numpy.add(x2, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.8651731))))),        numpy.multiply(lax_reference.div(1.0, numpy.add(numpy.multiply(2.0, numpy.subtract(1.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)))), 1e-07)), numpy.subtract(numpy.add(x2, 1.0), numpy.multiply(2.0, numpy.minimum(1.0, numpy.maximum(0.0, 0.674819)))))]],      dtype=object))), 0.5), numpy.array([[ True,  True,  True],       [ True,  True,  True]]), numpy.array([[False, False, False],       [False, False, False]]))'
    assert numpy.array_equal(
        symbolic_output,
        expected_output
    )
