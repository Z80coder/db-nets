import jax
import numpy
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

from neurallogic import hard_or, harden, neural_logic_net, symbolic_generation
from tests import utils


def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], 1.0],
        [[0.0, 0.0], [0.0, 0.0], 0.0],
        [[1.0, 0.0], [0.0, 1.0], 0.25],
        [[0.0, 1.0], [1.0, 0.0], 0.25],
        [[0.0, 1.0], [0.0, 0.0], 0.25],
        [[0.0, 1.0], [1.0, 1.0], 1.0],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_or.soft_or_neuron(weights, input)

        def hard(weights, input):
            return hard_or.hard_or_neuron(weights, input)

        utils.check_consistency(
            soft, hard, expected, jax.numpy.array(weights), jax.numpy.array(input)
        )


def test_layer():
    test_data = [
        [
            [1.0, 0.0],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]],
            [1., 0.25, 1., 0.25],
        ],
        [
            [1.0, 0.4],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [1., 0.47, 1., 0.25],
        ],
        [
            [0.0, 1.0],
            [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]],
            [1., 0.77, 0.25, 0.25],
        ],
        [
            [0.0, 0.0],
            [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.25, 0.25, 0.25, 0.],
        ],
    ]
    for input, weights, expected in test_data:

        def soft(weights, input):
            return hard_or.soft_or_layer(weights, input)

        def hard(weights, input):
            return hard_or.hard_or_layer(weights, input)

        utils.check_consistency(
            soft,
            hard,
            jax.numpy.array(expected),
            jax.numpy.array(weights),
            jax.numpy.array(input),
        )


def test_or():
    def test_net(type, x):
        x = hard_or.or_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = x.ravel()
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)

    test_data = [
        [[1.0, 1.0], [0.4877112, 0.45718065, 0.6026865, 0.95067847]],
        [[1.0, 0.0], [0.41678876, 0.27297217, 0.26314348, 0.57121617]],
        [[0.0, 1.0], [0.4877112, 0.45718065, 0.6026865, 0.95067847]],
        [[0.0, 0.0], [0.11372772, 0.09127802, 0.15657091, 0.23997489]],
    ]
    for input, expected in test_data:
        # Check that the soft function performs as expected
        output = soft.apply(weights, jax.numpy.array(input))
        print("output", output)
        print("expected", expected)
        assert jax.numpy.allclose(
            output, jax.numpy.array(expected)
        )

        # Check that the hard function performs as expected
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_output = hard.apply(hard_weights, hard_input)
        assert jax.numpy.allclose(hard_output, hard_expected)

        # Check that the symbolic function performs as expected
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert numpy.allclose(symbolic_output, hard_expected)


def test_train_or():
    def test_net(type, x):
        return hard_or.or_layer(type)(4, nn.initializers.uniform(1.0))(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])

    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    input = jax.numpy.array(x)
    output = jax.numpy.array(y)

    # Train the or layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(
        apply_fn=jax.vmap(soft.apply, in_axes=(None, 0)), params=weights, tx=tx
    )
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda params, x, y: jax.numpy.mean((state.apply_fn(params, x) - y) ** 2)
        )
    )
    for epoch in range(1, 500):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the and layer (both soft and hard variants) correctly predicts y
    weights = state.params
    hard_weights = harden.hard_weights(weights)

    for input, expected in zip(x, y):
        hard_input = harden.harden(jax.numpy.array(input))
        hard_expected = harden.harden(jax.numpy.array(expected))
        hard_result = hard.apply(hard_weights, hard_input)
        print("hard expected", hard_expected)
        print("hard result", hard_result)
        assert jax.numpy.allclose(hard_result, hard_expected)
        symbolic_output = symbolic.apply(hard_weights, hard_input)
        assert jax.numpy.array_equal(symbolic_output, hard_expected)


def test_symbolic_or():
    def test_net(type, x):
        x = hard_or.or_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = hard_or.or_layer(type)(4, nn.initializers.uniform(1.0))(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)

    # Compute soft result
    soft_input = jax.numpy.array([0.6, 0.45])
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
    symbolic_input = ["True", "False"]
    symbolic_weights = utils.make_symbolic(hard_weights)
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_input)
    symbolic_output = symbolic_generation.eval_symbolic_expression(symbolic_output)
    # Check that the symbolic result is the same as the hard result
    assert numpy.array_equal(symbolic_output, hard_result)

    # Compute symbolic result with symbolic inputs and non-symbolic weights
    symbolic_input = ["x1", "x2"]
    symbolic_output = symbolic.apply(hard_weights, symbolic_input)
    # Check the form of the symbolic expression
    assert numpy.array_equal(
        symbolic_output,
        [
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), False))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), False)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), False)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), False)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True)), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), True)), numpy.logical_and(lax_reference.ne(x2, 0), True)), 0), True))",
        ],
    )

    # Compute symbolic result with symbolic inputs and symbolic weights
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_input)
    # Check the form of the symbolic expression
    assert numpy.array_equal(
        symbolic_output,
        [
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(False, 0)))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))",
            "numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(False, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(False, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(numpy.logical_or(numpy.logical_or(False, numpy.logical_and(lax_reference.ne(x1, 0), lax_reference.ne(True, 0))), numpy.logical_and(lax_reference.ne(x2, 0), lax_reference.ne(True, 0))), 0), lax_reference.ne(True, 0)))",
        ],
    )
