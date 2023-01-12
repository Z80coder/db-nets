import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random
import numpy
import typing

from neurallogic import hard_and, harden, neural_logic_net, primitives, sym_gen, symbolic_primitives


def check_consistency(soft_function, hard_function, input, expected, soft_caller: typing.Callable, hard_caller: typing.Callable):
    # Check that the soft function performs as expected
    assert numpy.allclose(soft_caller(soft_function, input), expected)

    # Check that the hard function performs as expected
    #print(f'caller(hard_function, harden.harden(input))={hard_caller(hard_function, harden.harden(input))} with type {type(hard_caller(hard_function, harden.harden(input)))}')
    #print(f'harden.harden(expected)={harden.harden(expected)} with type {type(harden.harden(expected))}')
    assert numpy.allclose(hard_caller(hard_function, harden.harden(input)), harden.harden(expected))

    # Check that the soft and hard functions are consistent
    assert numpy.allclose(harden.harden(soft_caller(soft_function, input)), hard_caller(hard_function, harden.harden(input)))

    # Check that the symbolic version of the hard function performs as expected
    def make_symbolic_function(*input):
        return sym_gen.make_symbolic(hard_function, *input)
    symbolic_hard_function = hard_caller(
        make_symbolic_function, harden.harden(input))

    def make_eval_symbolic_function(*input):
        return sym_gen.eval_symbolic(symbolic_hard_function, *input)
    assert numpy.allclose(hard_caller(make_eval_symbolic_function, harden.harden(input)), harden.harden(expected))

    # Check that the symbolic version of the hard function, when evaluted with symbolic inputs, performs as expected
    symbolic_input = sym_gen.make_symbolic(numpy.array(input, dtype=object))

    def make_symbolic_expression_function(*input):
        return sym_gen.symbolic_expression(symbolic_hard_function, *input)
    symbolic_expression = hard_caller(
        make_symbolic_expression_function, symbolic_input)
    #print(f'symbolic_expression={symbolic_expression} with type {type(symbolic_expression)}')
    symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
    #print(f'symbolic_output={symbolic_output} with type {type(symbolic_output)}')
    #print(f'harden.harden(expected)={harden.harden(expected)} with type {type(harden.harden(expected))}')
    assert numpy.allclose(symbolic_output, harden.harden(expected))


def test_include():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 1.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 1.0]
    ]
    for input, expected in test_data:
        # Check that soft_and_include performs as expected
        assert hard_and.soft_and_include(*input) == expected

        # Check that hard_and_include performs as expected
        # assert hard_and.hard_and_include(*harden.harden(input)) == harden.harden(expected)
        hard_input = harden.harden(input)
        hard_output = hard_and.hard_and_include(*hard_input)
        hard_expected = harden.harden(expected)
        # assert hard_and.hard_and_include(*hard_input) == harden.harden(expected)
        assert hard_output == hard_expected

        # Check that soft_and_include and hard_and_include are consistent
        assert harden.harden(hard_and.soft_and_include(
            *input)) == hard_and.hard_and_include(*harden.harden(input))

        # Check that the symbolic version of hard_and_include performs as expected
        symbolic_hard_and_include = sym_gen.make_symbolic(
            hard_and.hard_and_include, *harden.harden(input))
        assert sym_gen.eval_symbolic(
            symbolic_hard_and_include, *harden.harden(input)) == harden.harden(expected)

        # Check that the symbolic version of hard_and_include, when evaluted with symbolic inputs, performs as expected
        symbolic_input = sym_gen.make_symbolic(
            numpy.array(input, dtype=object))
        symbolic_expression = sym_gen.symbolic_expression(
            symbolic_hard_and_include, *symbolic_input)
        symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
        assert symbolic_output == harden.harden(expected)

        # Do all the above again
        def soft_caller(f, x):
            return f(x[0], x[1])
        def hard_caller(f, x):
            return soft_caller(f, x)

        check_consistency(hard_and.soft_and_include,
                          hard_and.hard_and_include, input, expected, soft_caller, hard_caller)


def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], 1.0],
        [[0.0, 0.0], [0.0, 0.0], 1.0],
        [[1.0, 0.0], [0.0, 1.0], 0.0],
        [[0.0, 1.0], [1.0, 0.0], 0.0],
        [[0.0, 1.0], [0.0, 0.0], 1.0],
        [[0.0, 1.0], [1.0, 1.0], 0.0]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        assert jnp.allclose(hard_and.soft_and_neuron(weights, input), expected)
        assert jnp.allclose(hard_and.hard_and_neuron(harden.harden(
            weights), harden.harden(input)), harden.harden(expected))

        # Do all the above again
        hard_weights = harden.harden(weights)
        def soft_caller(f, x):
            return f(hard_weights, x)
        def hard_caller(f, x):
            return soft_caller(f, x)

        check_consistency(hard_and.soft_and_neuron,
                          hard_and.hard_and_neuron, input, expected, soft_caller, hard_caller)


def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
                      [0.0, 0.0]], [0.0, 0.0, 1.0, 1.0]],
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
                      [0.0, 0.2]], [0.0, 0.0, 1.0, 0.8]],
        [[1.0, 0.4], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
                      [0.0, 0.0]], [0.4, 0.4, 1.0, 1.0]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0],
                      [0.0, 0.0]], [0.0, 1.0, 0.0, 1.0]],
        [[0.0, 0.0], [[1.0, 0.01], [0.0, 1.0], [
            1.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0, 1.0]]
    ]
    for input, weights, expected in test_data:
        input = jnp.array(input)
        weights = jnp.array(weights)
        expected = jnp.array(expected)
        assert jnp.allclose(hard_and.soft_and_layer(weights, input), expected)
        assert jnp.allclose(hard_and.hard_and_layer(harden.harden(
            weights), harden.harden(input)), harden.harden(expected))

        # Do all the above again
        def soft_caller(f, x):
            return f(weights, x)
        hard_weights = harden.harden(weights)
        def hard_caller(f, x):
            return f(hard_weights, x)
        
        check_consistency(hard_and.soft_and_layer, hard_and.hard_and_layer, input, expected, soft_caller, hard_caller)


def test_and():
    def test_net(type, x):
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = primitives.nl_ravel(type)(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    test_data = [
        [
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
        [
            [1.0, 0.0],
            [0.53707814, 0.13892889, 0.7197356, 0.62835324]
        ],
        [
            [0.0, 1.0],
            [0.35097015, 0.34810758, 0.04889798, 0.43687034]
        ],
        [
            [0.0, 0.0],
            [0.35097015, 0.13892889, 0.04889798, 0.43687034]
        ]
    ]
    for input, expected in test_data:
        soft_input = jnp.array(input)
        soft_expected = jnp.array(expected)
        soft_result = soft.apply(soft_weights, soft_input)
        assert jnp.allclose(soft_result, soft_expected)
        hard_input = harden.harden(soft_input)
        hard_expected = harden.harden(soft_expected)
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.allclose(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)


def test_train_and():
    def test_net(type, x):
        return hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0]
    ]
    input = jnp.array(x)
    output = jnp.array(y)

    # Train the and layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(apply_fn=jax.vmap(
        soft.apply, in_axes=(None, 0)), params=soft_weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x,
                      y: jnp.mean((state.apply_fn(params, x) - y) ** 2)))
    for epoch in range(1, 100):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)

    # Test that the and layer (both soft and hard variants) correctly predicts y
    soft_weights = state.params
    hard_weights = harden.hard_weights(soft_weights)
    symbolic_weights = harden.symbolic_weights(soft_weights)
    for input, expected in zip(x, y):
        hard_input = harden.harden_array(harden.harden(jnp.array(input)))
        hard_expected = harden.harden_array(harden.harden(jnp.array(expected)))
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.allclose(hard_result, hard_expected)
        symbolic_result = symbolic.apply(symbolic_weights, hard_input.tolist())
        assert jnp.array_equal(symbolic_result, hard_expected)


def test_symbolic_and():
    def test_net(type, x):
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        return x

    soft, hard, symbolic = neural_logic_net.net(test_net)
    soft_weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    symbolic_weights = harden.symbolic_weights(soft_weights)
    symbolic_input = ['x1', 'x2']
    symbolic_result = symbolic.apply(symbolic_weights, symbolic_input)
    assert (symbolic_result == ['((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))', '((((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))',
            '((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(False)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(False)))', '((((x1 or not(True)) and (x2 or not(False))) or not(False)) and (((x1 or not(True)) and (x2 or not(True))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)) and (((x1 or not(True)) and (x2 or not(False))) or not(True)))'])
