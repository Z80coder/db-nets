import numpy

from neurallogic import symbolic_primitives

def test_binary_operator_str_str():
    output = symbolic_primitives.binary_operator("+", "1", "2")
    expected = "1 + 2"
    assert output == expected

def test_binary_operator_vector_vector():
    x1 = numpy.array(["1", "2"])
    x2 = numpy.array(["3", "4"])
    output = symbolic_primitives.binary_operator("+", x1, x2)
    expected = numpy.array(["1 + 3", "2 + 4"])
    assert numpy.all(output == expected)
    eval_output = symbolic_primitives.symbolic_eval(output)
    numpy_output = numpy.add(symbolic_primitives.symbolic_eval(x1), symbolic_primitives.symbolic_eval(x2))
    assert numpy.all(eval_output == numpy_output)

def test_binary_operator_matrix_vector():
    x1 = numpy.array([["1", "2"], ["3", "4"]])
    x2 = numpy.array(["5", "6"])
    output = symbolic_primitives.binary_operator("+", x1, x2)
    expected = numpy.array([["1 + 5", "2 + 6"], ["3 + 5", "4 + 6"]])
    assert numpy.all(output == expected)
    eval_output = symbolic_primitives.symbolic_eval(output)
    numpy_output = numpy.add(symbolic_primitives.symbolic_eval(x1), symbolic_primitives.symbolic_eval(x2))
    assert numpy.all(eval_output == numpy_output)

def test_binary_operator_vector_matrix():
    x1 = numpy.array(["1", "2"])
    x2 = numpy.array([["3", "4"], ["5", "6"]])
    output = symbolic_primitives.binary_operator("+", x1, x2)
    expected = numpy.array([["1 + 3", "2 + 4"], ["1 + 5", "2 + 6"]])
    assert numpy.all(output == expected)
    eval_output = symbolic_primitives.symbolic_eval(output)
    numpy_output = numpy.add(symbolic_primitives.symbolic_eval(x1), symbolic_primitives.symbolic_eval(x2))
    assert numpy.all(eval_output == numpy_output)

def test_binary_operator_matrix_matrix():
    x1 = numpy.array([["1", "2"], ["3", "4"]])
    x2 = numpy.array([["5", "6"], ["7", "8"]])
    output = symbolic_primitives.binary_operator("+", x1, x2)
    expected = numpy.array([["1 + 5", "2 + 6"], ["3 + 7", "4 + 8"]])
    assert numpy.all(output == expected)
    eval_output = symbolic_primitives.symbolic_eval(output)
    numpy_output = numpy.add(symbolic_primitives.symbolic_eval(x1), symbolic_primitives.symbolic_eval(x2))
    assert numpy.all(eval_output == numpy_output)

def test_binary_operator_matrix_matrix_2():
    # x1 is a (1,4) matrix
    x1 = numpy.array([["1", "2", "3", "4"]])
    # x2 is a (10, 4) matrix
    x2 = numpy.array([["5", "6", "7", "8"] for _ in range(10)])
    output = symbolic_primitives.binary_operator("+", x1, x2)
    expected = numpy.array([["1 + 5", "2 + 6", "3 + 7", "4 + 8"] for _ in range(10)])
    assert numpy.all(output == expected)
    eval_output = symbolic_primitives.symbolic_eval(output)
    numpy_output = numpy.add(symbolic_primitives.symbolic_eval(x1), symbolic_primitives.symbolic_eval(x2))
    assert numpy.all(eval_output == numpy_output)

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

def test_to_boolean_string():
    output = symbolic_primitives.to_boolean_string([1, 1])
    expected = numpy.array(["True", "True"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([0, 0])
    expected = numpy.array(["False", "False"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([True, False])
    expected = numpy.array(["True", "False"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([False, True])
    expected = numpy.array(["False", "True"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([1.0, 1.0])
    expected = numpy.array(["True", "True"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([0.0, 0.0])
    expected = numpy.array(["False", "False"])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([[1, 1], [1, 1]])
    expected = numpy.array([["True", "True"], ["True", "True"]])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([[0, 0], [0, 0]])
    expected = numpy.array([["False", "False"], ["False", "False"]])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([[True, False], [False, True]])
    expected = numpy.array([["True", "False"], ["False", "True"]])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([[[1, 0, 1], [1, 0, 1]], [[1, 0, 0], [1, 0, 0]]])
    expected = numpy.array([[["True", "False", "True"], ["True", "False", "True"]], [["True", "False", "False"], ["True", "False", "False"]]])
    assert numpy.all(output == expected)
    output = symbolic_primitives.to_boolean_string([[[1, "f", 1], [1, "g", 1]], [[1, "h", 0], [1, "f", 0]]])
    expected = numpy.array([[["True", "f", "True"], ["True", "g", "True"]], [["True", "h", "False"], ["True", "f", "False"]]])
    assert numpy.all(output == expected)
    
def test_symbolic_and():
    x1 = numpy.array([True, False])
    x2 = numpy.array([True, True])
    output = symbolic_primitives.symbolic_and(x1, x2)
    expected = numpy.array([True, False])
    assert numpy.all(output == expected)
