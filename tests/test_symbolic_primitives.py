import jax
import numpy

from neurallogic import symbolic_generation, symbolic_primitives, symbolic_operator
from tests import utils


def test_symbolic_expression():
    output = symbolic_operator.symbolic_operator("not", "True")
    expected = "not(True)"
    assert output == expected
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert eval_output == eval_expected


def test_symbolic_expression_vector():
    x = numpy.array(["True", "False"])
    output = symbolic_operator.symbolic_operator("not", x)
    expected = numpy.array(["not(True)", "not(False)"])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert numpy.array_equal(eval_output, eval_expected)


def test_symbolic_expression_matrix():
    x = numpy.array([["True", "False"], ["False", "True"]])
    output = symbolic_operator.symbolic_operator("not", x)
    expected = numpy.array(
        [["not(True)", "not(False)"], ["not(False)", "not(True)"]])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert numpy.array_equal(eval_output, eval_expected)



def test_symbolic_eval():
    output = symbolic_generation.eval_symbolic_expression("1 + 2")
    expected = 3
    assert output == expected
    output = symbolic_generation.eval_symbolic_expression("[1, 2, 3]")
    expected = [1, 2, 3]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(
        "[1, 2, 3] + [4, 5, 6]")
    expected = [1, 2, 3, 4, 5, 6]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(['1', '2', '3'])
    expected = [1, 2, 3]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(
        ['1', '2', '3'] + ['4', '5', '6'])
    expected = [1, 2, 3, 4, 5, 6]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(
        ['not(False)', 'not(True)'])
    expected = [True, False]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(
        [['not(False)', 'not(True)'] + ['not(False)', 'not(True)']])
    expected = [[True, False, True, False]]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(numpy.array(
        [['not(False)', 'not(True)'] + ['not(False)', 'not(True)']]))
    expected = [[True, False, True, False]]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(numpy.array(
        [['not(False)', False], ['not(False)', 'not(True)']]))
    expected = [[True, False], [True, False]]
    assert numpy.array_equal(output, expected)


def test_symbolic_not():
    x1 = numpy.array([True, False])
    output = symbolic_primitives.symbolic_not(x1)
    expected = numpy.array([False, True])
    assert numpy.array_equal(output, expected)
    x1 = utils.make_symbolic(x1)
    output = symbolic_primitives.symbolic_not(x1)
    expected = numpy.array(
        ['numpy.logical_not(True)', 'numpy.logical_not(False)'])
    assert numpy.array_equal(output, expected)


def test_symbolic_and():
    x1 = numpy.array([True, False])
    x2 = numpy.array([True, True])
    output = symbolic_primitives.symbolic_and(x1, x2)
    expected = numpy.array([True, False])
    assert numpy.array_equal(output, expected)
    x1 = utils.make_symbolic(x1)
    x2 = utils.make_symbolic(x2)
    output = symbolic_primitives.symbolic_and(x1, x2)
    expected = numpy.array(
        ['numpy.logical_and(True, True)', 'numpy.logical_and(False, True)'])
    assert numpy.array_equal(output, expected)


def test_symbolic_xor():
    x1 = numpy.array([True, False])
    x2 = numpy.array([True, True])
    output = symbolic_primitives.symbolic_xor(x1, x2)
    expected = numpy.array([False, True])
    assert numpy.array_equal(output, expected)
    x1 = utils.make_symbolic(x1)
    x2 = utils.make_symbolic(x2)
    output = symbolic_primitives.symbolic_xor(x1, x2)
    expected = numpy.array(
        ['numpy.logical_xor(True, True)', 'numpy.logical_xor(False, True)'])
    assert numpy.array_equal(output, expected)


def test_symbolic_broadcast_in_dim():
    # Test 1D
    input = jax.numpy.array([1, 1])
    output = symbolic_primitives.symbolic_broadcast_in_dim(input, (2, 2), (0,))
    expected = jax.numpy.array([[1, 1], [1, 1]])
    assert numpy.array_equal(output, expected)
    # Test 2D
    input = jax.numpy.array([[1, 1], [1, 1]])
    output = symbolic_primitives.symbolic_broadcast_in_dim(
        input, (2, 2, 2), (0, 1))
    expected = jax.numpy.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    assert numpy.array_equal(output, expected)
    # Test 3D
    input = jax.numpy.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    output = symbolic_primitives.symbolic_broadcast_in_dim(
        input, (2, 2, 2, 2), (0, 1, 2))
    expected = jax.numpy.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [
                               [[1, 1], [1, 1]], [[1, 1], [1, 1]]]])
    assert numpy.array_equal(output, expected)


def symbolic_reduce_or_impl(input, expected, symbolic_expected, axes):
    # symbolic_reduce_or uses the lax reference implementation if its input consists of boolean values,
    # otherwise it evaluates symbolically. Therefore we first test the reference implementation and then
    # the symbolic implementation, and then compare them.
    # Test reference implementation
    input = numpy.array(input)
    output = symbolic_primitives.symbolic_reduce_or(input, axes=axes)
    expected = numpy.array(expected)
    assert numpy.array_equal(output, expected)
    # Test symbolic implementation
    input = utils.make_symbolic(input)
    output = symbolic_primitives.symbolic_reduce_or(input, axes=axes)
    symbolic_expected = numpy.array(symbolic_expected)
    assert numpy.array_equal(output, symbolic_expected)
    # Compare the reference and symbolic evaluation
    symbolic_expected = symbolic_generation.eval_symbolic_expression(
        symbolic_expected)
    assert numpy.array_equal(expected, symbolic_expected)


def test_symbolic_reduce_or():
    # Test 1: 2D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=[
                            True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(False, True), False)', 'numpy.logical_or(numpy.logical_or(False, True), False)'], axes=(1,))
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=[
                            True, False], symbolic_expected=['numpy.logical_or(numpy.logical_or(False, True), True)', 'numpy.logical_or(numpy.logical_or(False, False), False)'], axes=(0,))
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=True,
                            symbolic_expected='numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False)', axes=(0, 1))
    # Test 2: 3D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False)',
                                                                                                                                              'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False)'], axes=(1, 2))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False)',
                                                                                                                                              'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False)'], axes=(0, 2))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, False], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), True), True), True)',
                                                                                                                                               'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, False), False), False), False)'], axes=(0, 1))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=True,
                            symbolic_expected='numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)', axes=(0, 1, 2))
    # Test 3: 4D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)',
                                                                                                                                                                                                                  'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)'], axes=(1, 2, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)',
                                                                                                                                                                                                                  'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)'], axes=(0, 2, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)',
                                                                                                                                                                                                                  'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False)'], axes=(0, 1, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, False], symbolic_expected=['numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), True), True), True), True), True), True), True)',
                                                                                                                                                                                                                   'numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, False), False), False), False), False), False), False), False)'], axes=(0, 1, 2))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=True,
                            symbolic_expected='numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(numpy.logical_or(False, True), False), True), False), True), False), True), False), True), False), True), False), True), False), True), False)', axes=(0, 1, 2, 3))
