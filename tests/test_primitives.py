import jax.numpy as jnp
import numpy
from neurallogic import primitives, neural_logic_net, harden
import pytest

# TODO: add symbolic tests

def test_ravel():
    input = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    expected = jnp.ravel(input)
    output = primitives.nl_ravel(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_ravel(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = jnp.asarray(harden.harden(input)).tolist()
    output = primitives.nl_ravel(neural_logic_net.NetType.Symbolic)(symbolic_input)
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)

def test_ravel_matrix():
    input = jnp.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.88]]])
    expected = jnp.ravel(input)
    output = primitives.nl_ravel(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_ravel(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = jnp.asarray(harden.harden(input)).tolist()
    output = primitives.nl_ravel(neural_logic_net.NetType.Symbolic)(symbolic_input)
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)

def test_ravel_symbolic():
    input = [[['x1', 'x2'], ['not(x1)', 'x1 | x2']], [['not(x2)', 'x1 & x2'], ['not(x1 & x2)', 'x2']]]
    expected = ['x1', 'x2', 'not(x1)', 'x1 | x2', 'not(x2)', 'x1 & x2', 'not(x1 & x2)', 'x2']
    output = primitives.nl_ravel(neural_logic_net.NetType.Symbolic)(input)
    assert output == expected

def test_sum():
    input = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    expected = jnp.sum(input, 1)
    output = primitives.nl_sum(neural_logic_net.NetType.Soft)(input, 1)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_sum(neural_logic_net.NetType.Hard)(input, 1)
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    output = primitives.nl_sum(neural_logic_net.NetType.Symbolic)(symbolic_input, 1)
    symbolic_expected = expected.tolist()
    assert output == symbolic_expected

def test_sum_matrix():
    input = jnp.array([[[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]]])
    expected = jnp.sum(input)
    output = primitives.nl_sum(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_sum(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    output = primitives.nl_sum(neural_logic_net.NetType.Symbolic)(symbolic_input)
    symbolic_expected = expected.tolist()
    assert output == expected

def test_sum_symbolic():
    input = [[['x1', 'x2'], ['not(x1)', 'x1 | x2']], [['not(x2)', 'x1 & x2'], ['not(x1 & x2)', 'x2']]]
    expected = '(((x1) + (not(x2))) + ((not(x1)) + (not(x1 & x2)))) + (((x2) + (x1 & x2)) + ((x1 | x2) + (x2)))'
    output = primitives.nl_sum(neural_logic_net.NetType.Symbolic)(input)
    assert output == expected

def test_reshape():
    input = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    expected = jnp.reshape(input, (1, 4))
    output = primitives.nl_reshape(neural_logic_net.NetType.Soft)(input, (1, 4))
    assert jnp.array_equal(output, expected)
    output = primitives.nl_reshape(neural_logic_net.NetType.Hard)(input, (1, 4))
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    output = primitives.nl_reshape(neural_logic_net.NetType.Symbolic)(symbolic_input, (1, 4))
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)
    assert numpy.shape(output) == expected.shape

def test_reshape_matrix():
    input = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
    expected = jnp.reshape(input, (1, 8))
    output = primitives.nl_reshape(neural_logic_net.NetType.Soft)(input, (1, 8))
    assert jnp.array_equal(output, expected)
    output = primitives.nl_reshape(neural_logic_net.NetType.Hard)(input, (1, 8))
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    output = primitives.nl_reshape(neural_logic_net.NetType.Symbolic)(symbolic_input, (1, 8))
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)
    assert numpy.shape(output) == expected.shape

@pytest.mark.skip(reason="mean not yet implemented")
def test_mean():
    input = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    expected = jnp.mean(input)
    output = primitives.nl_mean(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_mean(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    output = primitives.nl_mean(neural_logic_net.NetType.Symbolic)(symbolic_input)
    assert output == expected

@pytest.mark.skip(reason="mean not yet implemented")
def test_mean_matrix():
    input = jnp.array([[[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]]])
    expected = jnp.mean(input, -1)
    output = primitives.nl_mean(neural_logic_net.NetType.Soft)(input, -1)
    assert jnp.array_equal(output, expected)
    output = primitives.nl_mean(neural_logic_net.NetType.Hard)(input, -1)
    assert jnp.array_equal(output, expected)
    symbolic_input = harden.harden(input).tolist()
    symbolic_expected = expected.tolist()
    output = primitives.nl_mean(neural_logic_net.NetType.Symbolic)(symbolic_input, -1)
    print("output", output)
    print("expected", symbolic_expected)
    assert output == symbolic_expected

@pytest.mark.skip(reason="mean not yet implemented")
def test_mean_symbolic():
    input = [[['x1', 'x2'], ['not(x1)', 'x1 | x2']], [['not(x2)', 'x1 & x2'], ['not(x1 & x2)', 'x2']]]
    expected = '((((x1) + (not(x2))) + ((not(x1)) + (not(x1 & x2)))) + (((x2) + (x1 & x2)) + ((x1 | x2) + (x2)))) / 8'
    output = primitives.nl_mean(neural_logic_net.NetType.Symbolic)(input)
    assert output == expected