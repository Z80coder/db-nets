import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random
import numpy
import typing

from neurallogic import hard_and, harden, neural_logic_net, primitives, sym_gen, symbolic_primitives


def check_consistency(soft: typing.Callable, hard: typing.Callable,  symbolic: typing.Callable, expected, *args):
    # Check that the soft function performs as expected
    assert numpy.allclose(soft(*args), expected)

    # Check that the hard function performs as expected
    hard_args = harden.harden(*args)
    hard_expected = harden.harden(expected)
    assert numpy.allclose(hard(*hard_args), hard_expected)

    # Check that the symbolic function performs as expected
    symbolic_f = sym_gen.make_symbolic_jaxpr(symbolic, *hard_args)
    assert numpy.allclose(sym_gen.eval_symbolic(
        symbolic_f, *hard_args), hard_expected)

    # Check that the symbolic function, when evaluted with symbolic inputs, performs as expected
    symbolic_input = sym_gen.make_symbolic(*hard_args)
    symbolic_expression = sym_gen.symbolic_expression(
        symbolic_f, *symbolic_input)
    symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
    assert numpy.allclose(symbolic_output, hard_expected)


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
        check_consistency(hard_and.soft_and_include, hard_and.hard_and_include,
                          hard_and.hard_and_include, expected, input[0], input[1])


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
        def soft(weights, input):
            return hard_and.soft_and_neuron(weights, input)

        def hard(weights, input):
            return hard_and.hard_and_neuron(weights, input)

        def symbolic(weights, input):
            return hard(weights, input)

        check_consistency(soft, hard, symbolic, expected,
                          numpy.array(weights), numpy.array(input))


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
        def soft(weights, input):
            return hard_and.soft_and_layer(weights, input)

        def hard(weights, input):
            return hard_and.hard_and_layer(weights, input)

        def symbolic(weights, input):
            return hard(weights, input)

        check_consistency(soft, hard, symbolic, expected,
                          numpy.array(weights), numpy.array(input))


def test_and():
    def test_net(type, x):
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = x.ravel()
        return x

    soft, hard, jaxpr, symbolic = neural_logic_net.net(test_net)
    weights = soft.init(random.PRNGKey(0), [0.0, 0.0])
    hard_weights = harden.hard_weights(weights)

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
        # Check that the soft function performs as expected
        assert jnp.allclose(soft.apply(
            weights, jnp.array(input)), jnp.array(expected))

        # Check that the hard function performs as expected
        hard_input = harden.harden(jnp.array(input))
        hard_expected = harden.harden(jnp.array(expected))
        hard_output = hard.apply(hard_weights, hard_input)
        assert jnp.allclose(hard_output, hard_expected)

        # Check that the jaxpr expression performs as expected
        jaxpr_output = jaxpr.apply(hard_weights, hard_input)
        assert numpy.allclose(jaxpr_output, hard_expected)

        # Check that the symbolic function, when evaluted with symbolic inputs, performs as expected
        symbolic_expression = symbolic.apply(hard_weights, hard_input)
        symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
        assert numpy.allclose(symbolic_output, hard_expected)


def test_train_and():
    def test_net(type, x):
        return hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)

    soft, hard, jaxpr, symbolic = neural_logic_net.net(test_net)
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

    for input, expected in zip(x, y):
        hard_input = harden.harden_array(harden.harden(jnp.array(input)))
        hard_expected = harden.harden_array(harden.harden(jnp.array(expected)))
        hard_result = hard.apply(hard_weights, hard_input)
        assert jnp.allclose(hard_result, hard_expected)
        symbolic_expression = symbolic.apply(hard_weights, hard_input)
        symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
        assert jnp.array_equal(symbolic_output, hard_expected)


def test_symbolic_and():
    def test_net(type, x):
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        x = hard_and.and_layer(type)(4, nn.initializers.uniform(1.0))(x)
        return x

    soft, hard, jaxpr, symbolic = neural_logic_net.net(test_net)

    # Compute soft result
    soft_input = [0.6, 0.45]
    soft_weights = soft.init(random.PRNGKey(0), soft_input)
    soft_result = soft.apply(soft_weights, numpy.array(soft_input))
    
    # Compute hard result
    hard_weights = harden.hard_weights(soft_weights)
    hard_input = harden.harden(soft_input)
    hard_result = hard.apply(hard_weights, numpy.array(hard_input))
    # Check that the hard result is the same as the soft result
    assert numpy.array_equal(harden.harden(soft_result), hard_result)

    # Compute symbolic result with evaluable inputs
    symbolic_input = ['True', 'False']
    symbolic_expression = symbolic.apply(hard_weights, symbolic_input)
    symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
    # Check that the symbolic result is the same as the hard result
    assert numpy.array_equal(symbolic_output, hard_result)

    # Compute symbolic result with symbolic inputs
    symbolic_input = ['x1', 'x2']
    symbolic_expression = symbolic.apply(hard_weights, symbolic_input)
    # Check the form of the symbolic expression
    assert symbolic_expression.tolist() == ['True and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(True)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False))',
                                        'True and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(True)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False))',
                                        'True and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(True)) != 0.0 or not(False)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False))',
                                        'True and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(False)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(True)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True)) and (True and (x1 != 0.0 or not(True)) and (x2 != 0.0 or not(False)) != 0.0 or not(True))']

    # Compute symbolic result with symbolic inputs and symbolic weights
    #symbolic_weights = sym_gen.make_symbolic(hard_weights)
    #symbolic_expression = symbolic.apply(symbolic_weights, symbolic_input)
    #symbolic_output = sym_gen.eval_symbolic_expression(symbolic_expression)
    #print(f'symbolic_output = {symbolic_output}')
    # Check the form of the symbolic expression
    # TODO

