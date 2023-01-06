import numpy
from plum import dispatch
import jax
import jax._src.lax_reference as lax_reference

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

def all_boolean(data):
  if isinstance(data, bool):
    return True
  if isinstance(data, (list, tuple)):
    return all(all_boolean(x) for x in data)
  if isinstance(data, dict):
    return all(all_boolean(v) for v in data.values())
  if isinstance(data, numpy.ndarray):
    return all_boolean(data.tolist())
  if isinstance(data, jax.numpy.ndarray):
    return all_boolean(data.tolist())
  return False

def symbolic_and(*args, **kwargs):
  if all_boolean([*args]):
    return numpy.logical_and(*args, **kwargs)
  else:
    return binary_operator(" and ", *args, **kwargs)

def symbolic_xor(x, y):
  return f"{x} ^ {y}"

def symbolic_or(x, y):
  return f"{x} or {y}"

def symbolic_not(x):
  return f"~{x}"

# Uses the lax reference implementation of broadcast_in_dim to
# implement a symbolic version of broadcast_in_dim
def symbolic_broadcast_in_dim(*args, **kwargs):
  return lax_reference.broadcast_in_dim(*args, **kwargs)

# This function is a hack to get around the fact that JAX doesn't
# support symbolic reduction operations. It takes a symbolic reduction
# operation and a symbolic initial value and returns a function that
# performs the reduction operation on a numpy array.
def make_symbolic_reducer(py_binop, init_val):
  def reducer(operand, axis=0):
    # axis=0 means we are reducing over the first axis (i.e. the rows) of the operand.
    # axis=None means we are reducing over all axes of the operand.
    axis = range(numpy.ndim(operand)) if axis is None else axis

    # We create a new array with the same shape as the operand, but with the
    # dimensions corresponding to the axis argument removed. The values in this
    # array will be the result of the reduction.
    result = numpy.full(numpy.delete(numpy.shape(operand), axis), init_val, dtype=numpy.asarray(operand).dtype)

    # We iterate over all elements of the operand, computing the reduction.
    for idx, _ in numpy.ndenumerate(operand):
      # We need to index into the result array with the same indices that we used
      # to index into the operand, but with the axis dimensions removed.
      out_idx = tuple(numpy.delete(idx, axis))
      result[out_idx] = py_binop(result[out_idx], operand[idx])
    return result
  return reducer

def symbolic_reduce(operand, init_value, computation, dimensions):
  reducer = make_symbolic_reducer(computation, init_value)
  return reducer(operand, tuple(dimensions)).astype(operand.dtype)
  
def symbolic_reduce_or(*args, **kwargs):
  if all_boolean(*args):
    return lax_reference.reduce(*args, init_value=False, dimensions=kwargs['axes'], computation=numpy.logical_or)
  else:
    #print("args = ", args)
    #print("type args[0] = ", type(args[0]))
    #print("element type = ", args[0].dtype)
    r = symbolic_reduce(*args, init_value='False', dimensions=kwargs['axes'], computation=symbolic_or)
    #print("symbolic_reduce_or result: ", r)
    #print("type: ", type(r))
    #print("element type: ", r.dtype)
    return r

