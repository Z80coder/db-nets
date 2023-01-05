import numpy
from plum import dispatch
import jax

# to_boolean_value_string returns a string representation of a boolean value x, where
# x is either a boolean, 0, 1, the string 'True', or the string 'False'
def to_boolean_value_string(x):
    if isinstance(x, bool):
        # x is a bool
        return 'True' if x else 'False'
    elif x == 1.0 or x == 0.0:
        # x is a float
        return 'True' if x == 1 else 'False'
    elif isinstance(x, str) and (x == '1' or x == '0'):
        # x is a string representing an integer
        return 'True' if x == '1' else 'False'
    elif isinstance(x, str) and (x == '1.0' or x == '0.0'):
        # x is a string representing a float
        return 'True' if x == '1.0' else 'False'
    elif isinstance(x, str) and (x == 'True' or x == 'False'):
        # x is a string representing a boolean
        return 'True' if x == 'True' else 'False'
    else:
        # x is not interpretable as a boolean
        return str(x)

def to_boolean_string(x):
    """Converts an arbitrary vector of arbitrary values to a numpy array where
    every boolean-interpretable value gets converted to the strings "True" or "False".

    Args:
        x: The vector of values to convert (or can be a single value in the degenerate case)

    Returns:
        A numpy array representation of the input, where boolean-interpretable
        values are converted to "True" or "False".
    """
    if isinstance(x, numpy.ndarray) or isinstance(x, jax.numpy.ndarray):
        return to_boolean_string(x.tolist())
    elif isinstance(x, list):
        return numpy.array([to_boolean_string(y) for y in x], dtype=object)
    else:
        return to_boolean_value_string(x)

@dispatch
def binary_operator(operator: str, a: str, b: str) -> str:
    return f"{a} {operator} {b}"

@dispatch
def binary_operator(operator: str, a: numpy.ndarray, b: numpy.ndarray):
    #print("binary_operator", operator, a, b)
    #print("element type", a.dtype, b.dtype)
    r = numpy.vectorize(binary_operator, otypes=[object])(operator, a, b)
    #print("r", r)
    #print("element type", r.dtype)
    return r

def symbolic_eval(x):
    return numpy.vectorize(eval)(x)

def is_boolean_value(x):
    return x.dtype == bool

def symbolic_and(*args, **kwargs):
  if is_boolean_value(args[0]):
    return numpy.logical_and(*args, **kwargs)
  else:
    return binary_operator(" and ", *args, **kwargs)

def symbolic_xor(x, y):
  return f"{x} ^ {y}"

def symbolic_or(x, y):
  return f"{x} or {y}"

def symbolic_not(x):
  return f"~{x}"
