import numpy
from plum import dispatch
import typing
import jax
import jax._src.lax_reference as lax_reference
import jaxlib


def convert_element_type(x, dtype):
    if dtype == numpy.int32 or dtype == numpy.int64:
        dtype = "int"
    elif dtype == bool:
        dtype = "bool"
    else:
        raise NotImplementedError(
            f"Symbolic conversion to {dtype} is not implemented")

    def convert(x):
        return f"{dtype}({x})"
    return map_at_elements(x, convert)


def convert_iterable_type(x: list, new_type):
    if new_type == list:
        return x
    elif new_type == numpy.ndarray:
        return numpy.array(x, dtype=object)
    elif new_type == jax.numpy.ndarray:
        return jax.numpy.array(x, dtype=object)
    elif new_type == jaxlib.xla_extension.DeviceArray:
        return jax.numpy.array(x, dtype=object)
    else:
        raise NotImplementedError(
            f"Cannot convert type {type(x)} to type {new_type}")


@dispatch
def map_at_elements(x: list, func: typing.Callable):
    return convert_iterable_type([map_at_elements(item, func) for item in x], type(x))


@dispatch
def map_at_elements(x: numpy.ndarray, func: typing.Callable):
    return convert_iterable_type([map_at_elements(item, func) for item in x], type(x))


@dispatch
def map_at_elements(x: jax.numpy.ndarray, func: typing.Callable):
    if x.ndim == 0:
        return func(x.item())
    return convert_iterable_type([map_at_elements(item, func) for item in x], type(x))


@dispatch
def map_at_elements(x: str, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x, func: typing.Callable):
    return func(x)


"""
@dispatch
def to_boolean_value_string(x: bool):
    return '1' if x else '0'


@dispatch
def to_boolean_value_string(x: numpy.bool_):
    return '1' if x else '0'


@dispatch
def to_boolean_value_string(x: int):
    return '1' if x >= 1 else '0'


@dispatch
def to_boolean_value_string(x: float):
    return '1' if x >= 1.0 else '0'


@dispatch
def to_boolean_value_string(x: str):
    if x == '1' or x == '1.0' or x == 'True':
        return '1'
    elif x == '0' or x == '0.0' or x == 'False':
        return '0'
    else:
        return x
"""


@dispatch
def to_boolean_value_string(x: bool):
    return 'True' if x else 'False'


@dispatch
def to_boolean_value_string(x: numpy.bool_):
    return 'True' if x else 'False'


@dispatch
def to_boolean_value_string(x: int):
    return 'True' if x >= 1 else 'False'


@dispatch
def to_boolean_value_string(x: float):
    return 'True' if x >= 1.0 else 'False'


@dispatch
def to_boolean_value_string(x: str):
    if x == '1' or x == '1.0' or x == 'True':
        return 'True'
    elif x == '0' or x == '0.0' or x == 'False':
        return 'False'
    else:
        return x



@dispatch
def unary_operator(operator: str, x: str) -> str:
    return f"{operator}({x})"


@dispatch
def unary_operator(operator: str, x: numpy.ndarray):
    return numpy.vectorize(unary_operator, otypes=[object])(operator, x)


@dispatch
def unary_operator(operator: str, x: list):
    return unary_operator(operator, numpy.array(x))


@dispatch
def binary_infix_operator(operator: str, a: str, b: str, bracket: bool = False) -> str:
    if bracket:
        return f"({a} {operator} {b})"
    return f"{a} {operator} {b}"


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: numpy.ndarray, bracket: bool = False):
    return numpy.vectorize(binary_infix_operator, otypes=[object])(operator, a, b, bracket)


@dispatch
def binary_infix_operator(operator: str, a: list, b: numpy.ndarray, bracket: bool = False):
    return binary_infix_operator(operator, numpy.array(a), b, bracket)


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: list, bracket: bool = False):
    return binary_infix_operator(operator, a, numpy.array(b), bracket)


@dispatch
def binary_infix_operator(operator: str, a: str, b: int, bracket: bool = False):
    return binary_infix_operator(operator, a, str(b), bracket)


def all_concrete_values(data):
    if isinstance(data, str):
        return False
    if isinstance(data, (list, tuple)):
        return all(all_concrete_values(x) for x in data)
    if isinstance(data, dict):
        return all(all_concrete_values(v) for v in data.values())
    if isinstance(data, numpy.ndarray):
        return all_concrete_values(data.tolist())
    if isinstance(data, jax.numpy.ndarray):
        return all_concrete_values(data.tolist())
    return True


def symbolic_not(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_not(*args, **kwargs)
    else:
        return unary_operator("not", *args, **kwargs)


def symbolic_ne(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.not_equal(*args, **kwargs)
    else:
        return binary_infix_operator("!=", *args, **kwargs)


def symbolic_and(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_and(*args, **kwargs)
    else:
        return binary_infix_operator("and", *args, **kwargs)


def symbolic_or(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_or(*args, **kwargs)
    else:
        return binary_infix_operator("or", *args, **kwargs, bracket=True)


def symbolic_xor(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_xor(*args, **kwargs)
    else:
        return binary_infix_operator("^", *args, **kwargs)



def symbolic_sum(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.sum(*args, **kwargs)
    else:
        return binary_infix_operator("+", *args, **kwargs)


def symbolic_broadcast_in_dim(*args, **kwargs):
    # Uses the lax reference implementation of broadcast_in_dim to
    # implement a symbolic version of broadcast_in_dim
    return lax_reference.broadcast_in_dim(*args, **kwargs)


# TODO: add a test for this
def symbolic_convert_element_type(*args, **kwargs):
    # Check if all the boolean arguments are True or False
    if all_concrete_values([*args]):
        # If so, we can use the lax reference implementation
        return lax_reference.convert_element_type(*args, dtype=kwargs['new_dtype'])
    else:
        # Otherwise, we use the symbolic implementation
        return convert_element_type(*args, dtype=kwargs['new_dtype'])


def make_symbolic_reducer(py_binop, init_val):
    # This function is a hack to get around the fact that JAX doesn't
    # support symbolic reduction operations. It takes a symbolic reduction
    # operation and a symbolic initial value and returns a function that
    # performs the reduction operation on a numpy array.
    def reducer(operand, axis):
        # axis=None means we are reducing over all axes of the operand.
        axis = range(numpy.ndim(operand)) if axis is None else axis

        # We create a new array with the same shape as the operand, but with the
        # dimensions corresponding to the axis argument removed. The values in this
        # array will be the result of the reduction.
        result = numpy.full(numpy.delete(numpy.shape(
            operand), axis), init_val, dtype=numpy.asarray(operand).dtype)

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


def symbolic_reduce_and(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.reduce(*args, init_value=True, computation=numpy.logical_and, dimensions=kwargs['axes'])
    else:
        return symbolic_reduce(*args, init_value='True', computation=symbolic_and, dimensions=kwargs['axes'])


def symbolic_reduce_or(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.reduce(*args, init_value=False, computation=numpy.logical_or, dimensions=kwargs['axes'])
    else:
        return symbolic_reduce(*args, init_value='False', computation=symbolic_or, dimensions=kwargs['axes'])


def symbolic_reduce_sum(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.reduce(*args, init_value=0, computation=numpy.add, dimensions=kwargs['axes'])
    else:
        return symbolic_reduce(*args, init_value='0', computation=symbolic_sum, dimensions=kwargs['axes'])
