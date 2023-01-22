import jax
import numpy
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

from neurallogic import hard_xor, harden, neural_logic_net, symbolic_generation
from tests import utils


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
        utils.check_consistency(hard_xor.soft_xor_include, hard_xor.hard_xor_include,
                                expected, input[0], input[1])


def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], 0.0],
        [[0.0, 0.0], [1.0, 1.0], 0.0],
        [[1.0, 0.0], [1.0, 1.0], 1.0],
        [[0.0, 1.0], [1.0, 1.0], 1.0],

        [[1.0, 1.0], [0.0, 1.0], 1.0],
        [[0.0, 0.0], [0.0, 1.0], 0.0],
        [[1.0, 0.0], [0.0, 1.0], 0.0],
        [[0.0, 1.0], [0.0, 1.0], 1.0],

        [[1.0, 1.0], [1.0, 0.0], 1.0],
        [[0.0, 0.0], [1.0, 0.0], 0.0],
        [[1.0, 0.0], [1.0, 0.0], 1.0],
        [[0.0, 1.0], [1.0, 0.0], 0.0],

        [[1.0, 1.0], [0.0, 0.0], 0.0],
        [[0.0, 0.0], [0.0, 0.0], 0.0],
        [[1.0, 0.0], [0.0, 0.0], 0.0],
        [[0.0, 1.0], [0.0, 0.0], 0.0]
    ]
    for input, weights, expected in test_data:
        def soft(weights, input):
            return hard_xor.soft_xor_neuron(weights, input)

        def hard(weights, input):
            return hard_xor.hard_xor_neuron(weights, input)

        utils.check_consistency(soft, hard, expected,
                                jax.numpy.array(weights), jax.numpy.array(input))


def test_layer():
    test_data = [
        [
            [1.0, 0.0],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]],
            [1.0, 0.0, 1.0, 0.0]
        ],
        [
            [1.0, 0.4],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.6, 0.39999998, 1.0, 0.0]
        ],
        [
            [0.0, 1.0],
            [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]],
            [1.0, 0.8, 0.0, 0.0]
        ],
        [
            [0.0, 0.0],
            [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.8, 0.6],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.39999998, 0.6, 0.8, 0.0]
        ]
    ]
    for input, weights, expected in test_data:
        def soft(weights, input):
            return hard_xor.soft_xor_layer(weights, input)

        def hard(weights, input):
            return hard_xor.hard_xor_layer(weights, input)

        utils.check_consistency(soft, hard, jax.numpy.array(expected),
                                jax.numpy.array(weights), jax.numpy.array(input))


def test_xor():
    def test_net(type, x):
        x = hard_xor.xor_layer(type)(4)(x)
        x = x.ravel()
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)

    test_data = [
        [
            [1.0, 1.0],
            [0.56627953, 0.5280515, 0.18560958, 0.7154212]
        ],
        [
            [1.0, 0.0],
            [0.38767397, 0.99864066, 0.8143904, 0.17714691]
        ],
        [
            [0.0, 1.0],
            [0.56627953, 0.4719485, 0.8841522, 0.7154212]
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        soft_output = soft.apply(weights, jax.numpy.array(input))
        soft_expected = jax.numpy.array(expected)
        assert jax.numpy.allclose(soft_output, soft_expected)

        # Check that the hard function performs as expected
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, hard_input)
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert numpy.allclose(symbolic_output, hard_expected)


def test_train_xor():
    def test_net(type, x):
        return hard_xor.xor_layer(type)(4)(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])

    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]
    input = jax.numpy.array(x)
    output = jax.numpy.array(y)

    # Train the xor layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(apply_fn=jax.vmap(
        soft.apply, in_axes=(None, 0)), params=weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x,
                      y: jax.numpy.mean((state.apply_fn(params, x) - y) ** 2)))
    for epoch in range(1, 100):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the and layer (both soft and hard variants) correctly predicts y
    weights = state.params
    hard_weights = harden.hard_weights(weights)

    for input, expected in zip(x, y):
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_result = hard.apply(hard_weights, hard_input)
        assert jax.numpy.allclose(hard_result, hard_expected)
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert jax.numpy.array_equal(symbolic_output, hard_expected)


def test_symbolic_xor():
    def test_net(type, x):
        x = hard_xor.xor_layer(type)(4)(x)
        x = hard_xor.xor_layer(type)(4)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)

    # Compute soft result
    soft_input = jax.numpy.array([0.6, 0.55])
    weights = soft.init(random.PRNGKey(0), soft_input)
    soft_result = soft.apply(weights, numpy.array(soft_input))

    # Compute hard result
    hard_weights = harden.hard_weights(weights)
    hard_input = harden.harden(soft_input)
    hard_result = hard.apply(hard_weights, numpy.array(hard_input))
    # Check that the hard result is the same as the soft result
    assert numpy.array_equal(harden.harden(soft_result), hard_result)

    # Compute symbolic result with non-symbolic inputs
    symbolic_output = symbolic.apply(hard_weights, hard_input)
    # Check that the symbolic result is the same as the hard result
    assert numpy.array_equal(symbolic_output, hard_result)

    # Compute symbolic result with symbolic inputs and symbolic weights, but where the symbols can be evaluated
    symbolic_input = ['True', 'True']
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
    assert numpy.array_equal(symbolic_output, ['numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))'])

    # Compute symbolic result with symbolic inputs and symbolic weights
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_input)
    # Check the form of the symbolic expression
    assert numpy.array_equal(symbolic_output, ['numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0)))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))',
                                               'numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_xor(numpy.logical_xor(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))'])
