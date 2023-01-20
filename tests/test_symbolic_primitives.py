import numpy
import jax
from neurallogic import symbolic_primitives, symbolic_generation


def test_unary_operator_str():
    output = symbolic_primitives.symbolic_operator("not", "True")
    expected = "not(True)"
    assert output == expected
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert eval_output == eval_expected


def test_unary_operator_vector():
    x = numpy.array(["True", "False"])
    output = symbolic_primitives.symbolic_operator("not", x)
    expected = numpy.array(["not(True)", "not(False)"])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert numpy.array_equal(eval_output, eval_expected)


def test_unary_operator_matrix():
    x = numpy.array([["True", "False"], ["False", "True"]])
    output = symbolic_primitives.symbolic_operator("not", x)
    expected = numpy.array(
        [["not(True)", "not(False)"], ["not(False)", "not(True)"]])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    eval_expected = symbolic_generation.eval_symbolic_expression(expected)
    assert numpy.array_equal(eval_output, eval_expected)


def test_binary_operator_str_str():
    output = symbolic_primitives.symbolic_infix_operator("+", "1", "2")
    expected = "1 + 2"
    assert output == expected
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        "1"), symbolic_generation.eval_symbolic_expression("2"))
    assert numpy.array_equal(eval_output, numpy_output)


def test_binary_operator_vector_vector():
    x1 = numpy.array(["1", "2"])
    x2 = numpy.array(["3", "4"])
    output = symbolic_primitives.symbolic_infix_operator("+", x1, x2)
    expected = numpy.array(["1 + 3", "2 + 4"])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        x1), symbolic_generation.eval_symbolic_expression(x2))
    assert numpy.array_equal(eval_output, numpy_output)


def test_binary_operator_matrix_vector():
    x1 = numpy.array([["1", "2"], ["3", "4"]])
    x2 = numpy.array(["5", "6"])
    output = symbolic_primitives.symbolic_infix_operator("+", x1, x2)
    expected = numpy.array([["1 + 5", "2 + 6"], ["3 + 5", "4 + 6"]])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        x1), symbolic_generation.eval_symbolic_expression(x2))
    assert numpy.array_equal(eval_output, numpy_output)


def test_binary_operator_vector_matrix():
    x1 = numpy.array(["1", "2"])
    x2 = numpy.array([["3", "4"], ["5", "6"]])
    output = symbolic_primitives.symbolic_infix_operator("+", x1, x2)
    expected = numpy.array([["1 + 3", "2 + 4"], ["1 + 5", "2 + 6"]])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        x1), symbolic_generation.eval_symbolic_expression(x2))
    assert numpy.array_equal(eval_output, numpy_output)


def test_binary_operator_matrix_matrix():
    x1 = numpy.array([["1", "2"], ["3", "4"]])
    x2 = numpy.array([["5", "6"], ["7", "8"]])
    output = symbolic_primitives.symbolic_infix_operator("+", x1, x2)
    expected = numpy.array([["1 + 5", "2 + 6"], ["3 + 7", "4 + 8"]])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        x1), symbolic_generation.eval_symbolic_expression(x2))
    assert numpy.array_equal(eval_output, numpy_output)


def test_binary_operator_matrix_matrix_2():
    # x1 is a (1,4) matrix
    x1 = numpy.array([["1", "2", "3", "4"]])
    # x2 is a (10, 4) matrix
    x2 = numpy.array([["5", "6", "7", "8"] for _ in range(10)])
    output = symbolic_primitives.symbolic_infix_operator("+", x1, x2)
    expected = numpy.array(
        [["1 + 5", "2 + 6", "3 + 7", "4 + 8"] for _ in range(10)])
    assert numpy.array_equal(output, expected)
    eval_output = symbolic_generation.eval_symbolic_expression(output)
    numpy_output = numpy.add(symbolic_generation.eval_symbolic_expression(
        x1), symbolic_generation.eval_symbolic_expression(x2))
    assert numpy.array_equal(eval_output, numpy_output)


def test_to_boolean_value_string():
    output = symbolic_primitives.to_boolean_value_string(1)
    expected = "True"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string(0)
    expected = "False"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("1")
    expected = "True"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("0")
    expected = "False"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string(1.0)
    expected = "True"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string(0.0)
    expected = "False"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("1.0")
    expected = "True"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("0.0")
    expected = "False"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("True")
    expected = "True"
    assert output == expected
    output = symbolic_primitives.to_boolean_value_string("False")
    expected = "False"
    assert output == expected



def test_symbolic_eval():
    output = symbolic_generation.eval_symbolic_expression("1 + 2")
    expected = 3
    assert output == expected
    output = symbolic_generation.eval_symbolic_expression("[1, 2, 3]")
    expected = [1, 2, 3]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression("[1, 2, 3] + [4, 5, 6]")
    expected = [1, 2, 3, 4, 5, 6]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(['1', '2', '3'])
    expected = [1, 2, 3]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(
        ['1', '2', '3'] + ['4', '5', '6'])
    expected = [1, 2, 3, 4, 5, 6]
    assert numpy.array_equal(output, expected)
    output = symbolic_generation.eval_symbolic_expression(['not(False)', 'not(True)'])
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
    x1 = symbolic_generation.make_symbolic(x1)
    output = symbolic_primitives.symbolic_not(x1)
    expected = numpy.array(["not(True)", "not(False)"])
    assert numpy.array_equal(output, expected)


def test_symbolic_and():
    x1 = numpy.array([True, False])
    x2 = numpy.array([True, True])
    output = symbolic_primitives.symbolic_and(x1, x2)
    expected = numpy.array([True, False])
    assert numpy.array_equal(output, expected)
    x1 = symbolic_generation.make_symbolic(x1)
    x2 = symbolic_generation.make_symbolic(x2)
    output = symbolic_primitives.symbolic_and(x1, x2)
    expected = numpy.array(["True and True", "False and True"])
    assert numpy.array_equal(output, expected)


def test_symbolic_xor():
    x1 = numpy.array([True, False])
    x2 = numpy.array([True, True])
    output = symbolic_primitives.symbolic_xor(x1, x2)
    expected = numpy.array([False, True])
    assert numpy.array_equal(output, expected)
    x1 = symbolic_generation.make_symbolic(x1)
    x2 = symbolic_generation.make_symbolic(x2)
    output = symbolic_primitives.symbolic_xor(x1, x2)
    expected = numpy.array(["True ^ True", "False ^ True"])
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
    input = symbolic_generation.make_symbolic(input)
    output = symbolic_primitives.symbolic_reduce_or(input, axes=axes)
    symbolic_expected = numpy.array(symbolic_expected)
    assert numpy.array_equal(output, symbolic_expected)
    # Compare the reference and symbolic evaluation
    symbolic_expected = symbolic_generation.eval_symbolic_expression(symbolic_expected)
    assert numpy.array_equal(expected, symbolic_expected)


def test_symbolic_reduce_or():
    # Test 1: 2D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=[
                            True, True], symbolic_expected=['((False or True) or False)', '((False or True) or False)'], axes=(1,))
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=[
                            True, False], symbolic_expected=['((False or True) or True)', '((False or False) or False)'], axes=(0,))
    symbolic_reduce_or_impl(input=[[True, False], [True, False]], expected=True,
                            symbolic_expected='((((False or True) or False) or True) or False)', axes=(0, 1))
    # Test 2: 3D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, True], symbolic_expected=
                            ['((((False or True) or False) or True) or False)', '((((False or True) or False) or True) or False)'], axes=(1, 2))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, True], symbolic_expected=
                            ['((((False or True) or False) or True) or False)', '((((False or True) or False) or True) or False)'], axes=(0, 2))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=[True, False], symbolic_expected=
                            ['((((False or True) or True) or True) or True)', '((((False or False) or False) or False) or False)'], axes=(0, 1))
    symbolic_reduce_or_impl(input=[[[True, False], [True, False]], [[True, False], [True, False]]], expected=True,
                            symbolic_expected='((((((((False or True) or False) or True) or False) or True) or False) or True) or False)', axes=(0, 1, 2))
    # Test 3: 4D matrix with different axes inputs
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=
                            ['((((((((False or True) or False) or True) or False) or True) or False) or True) or False)', '((((((((False or True) or False) or True) or False) or True) or False) or True) or False)'], axes=(1, 2, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=
                            ['((((((((False or True) or False) or True) or False) or True) or False) or True) or False)', '((((((((False or True) or False) or True) or False) or True) or False) or True) or False)'], axes=(0, 2, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, True], symbolic_expected=
                            ['((((((((False or True) or False) or True) or False) or True) or False) or True) or False)', '((((((((False or True) or False) or True) or False) or True) or False) or True) or False)'], axes=(0, 1, 3))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=[True, False], symbolic_expected=
                            ['((((((((False or True) or True) or True) or True) or True) or True) or True) or True)', '((((((((False or False) or False) or False) or False) or False) or False) or False) or False)'], axes=(0, 1, 2))
    symbolic_reduce_or_impl(input=[[[[True, False], [True, False]], [[True, False], [True, False]]], [[[True, False], [True, False]], [[True, False], [True, False]]]], expected=True,
                            symbolic_expected='((((((((((((((((False or True) or False) or True) or False) or True) or False) or True) or False) or True) or False) or True) or False) or True) or False) or True) or False)', axes=(0, 1, 2, 3))
