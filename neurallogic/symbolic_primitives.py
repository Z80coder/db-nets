import numpy
from plum import dispatch
import typing
import jax
import jax._src.lax_reference as lax_reference
import jaxlib


def convert_element_type(x, dtype):
    if dtype == numpy.int32 or dtype == numpy.int64:
        dtype = 'int'
    elif dtype == bool:
        dtype = 'bool'
    elif dtype == numpy.float32:
        dtype = 'float'
    else:
        raise NotImplementedError(
            f'Symbolic conversion of type {type(x)} to {dtype} not implemented'
        )

    def convert(x):
        return f'{dtype}({x})'

    return map_at_elements(x, convert)


# TODO: allow func callable to control the type of the numpy.array or jax.numpy.array

# map_at_elements should alter the elements but not the type of the container


@dispatch
def map_at_elements(x: str, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: bool, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: numpy.bool_, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: float, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: list, func: typing.Callable):
    return [map_at_elements(item, func) for item in x]


@dispatch
def map_at_elements(x: numpy.ndarray, func: typing.Callable):
    return numpy.array([map_at_elements(item, func) for item in x], dtype=object)


@dispatch
def map_at_elements(x: jax.numpy.ndarray, func: typing.Callable):
    if x.ndim == 0:
        return func(x.item())
    return jax.numpy.array([map_at_elements(item, func) for item in x])


@dispatch
def map_at_elements(x: dict, func: typing.Callable):
    return {k: map_at_elements(v, func) for k, v in x.items()}


@dispatch
def map_at_elements(x: tuple, func: typing.Callable):
    return tuple(map_at_elements(list(x), func))


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
def to_numeric_value(x):
    if x == 'True' or x:
        return 1
    elif x == 'False' or not x:
        return 0
    elif isinstance(x, int) or isinstance(x, float):
        return x
    else:
        return 0


@dispatch
def unary_operator(operator: str, x: str) -> str:
    return f'{operator}({x})'


@dispatch
def unary_operator(operator: str, x: numpy.ndarray):
    return numpy.vectorize(unary_operator, otypes=[object])(operator, x)


@dispatch
def unary_operator(operator: str, x: list):
    return unary_operator(operator, numpy.array(x))


@dispatch
def binary_infix_operator(operator: str, a: str, b: str) -> str:
    return f'{a} {operator} {b}'


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: numpy.ndarray):
    return numpy.vectorize(binary_infix_operator, otypes=[object])(operator, a, b)


@dispatch
def binary_infix_operator(operator: str, a: list, b: numpy.ndarray):
    return binary_infix_operator(operator, numpy.array(a), b)


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: list):
    return binary_infix_operator(operator, a, numpy.array(b))


@dispatch
def binary_infix_operator(operator: str, a: str, b: int):
    return binary_infix_operator(operator, a, str(b))


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: float):
    return binary_infix_operator(operator, a, str(b))


@dispatch
def binary_infix_operator(operator: str, a: str, b: float):
    return binary_infix_operator(operator, a, str(b))


@dispatch
def binary_infix_operator(operator: str, a: numpy.ndarray, b: jax.numpy.ndarray):
    return binary_infix_operator(operator, a, numpy.array(b))


@dispatch
def binary_infix_operator(operator: str, a: bool, b: str):
    return binary_infix_operator(operator, str(a), b)


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
        return unary_operator('not', *args, **kwargs)


def symbolic_eq(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.eq(*args, **kwargs)
    else:
        return '(' + binary_infix_operator('==', *args, **kwargs) + ')'


def symbolic_ne(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.ne(*args, **kwargs)
    else:
        return '(' + binary_infix_operator('!=', *args, **kwargs) + ')'


def symbolic_le(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.le(*args, **kwargs)
    else:
        return '(' + binary_infix_operator('<=', *args, **kwargs) + ')'


def symbolic_lt(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.lt(*args, **kwargs)
    else:
        return '(' + binary_infix_operator('<', *args, **kwargs) + ')'


def symbolic_gt(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.gt(*args, **kwargs)
    else:
        return binary_infix_operator('>', *args, **kwargs)


def symbolic_add(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.add(*args, **kwargs)
    else:
        return unary_operator('np.add', *args, **kwargs)


def symbolic_sub(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.sub(*args, **kwargs)
    else:
        return unary_operator('np.subtract', *args, **kwargs)


def symbolic_mul(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.mul(*args, **kwargs)
    else:
        return unary_operator('np.multiply', *args, **kwargs)


def symbolic_div(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.div(*args, **kwargs)
    else:
        return unary_operator('div', *args, **kwargs)


def symbolic_max(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.max(*args, **kwargs)
    else:
        return unary_operator('np.maximum', *args, **kwargs)


def symbolic_min(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.min(*args, **kwargs)
    else:
        return unary_operator('np.minimum', *args, **kwargs)


def symbolic_select_n(*args, **kwargs):
    '''
    Important comment from lax.py
    # Caution! The select_n_p primitive has the *opposite* order of arguments to
    # select(). This is because it implements `select_n`.
    '''
    pred = args[0]
    on_true = args[1]
    on_false = args[2]
    if all_concrete_values([*args]):
        # swap order of on_true and on_false
        return lax_reference.select(pred, on_false, on_true)
    else:
        # swap order of on_true and on_false
        return unary_operator('np.where', pred, on_false, on_true)


def symbolic_and(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_and(*args, **kwargs)
    else:
        return binary_infix_operator('and', *args, **kwargs)


def symbolic_or(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_or(*args, **kwargs)
    else:
        return '(' + binary_infix_operator('or', *args, **kwargs) + ')'


def symbolic_xor(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_xor(*args, **kwargs)
    else:
        return binary_infix_operator('^', *args, **kwargs)


def symbolic_sum(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.sum(*args, **kwargs)
    else:
        return binary_infix_operator('+', *args, **kwargs)


def symbolic_broadcast_in_dim(*args, **kwargs):
    return lax_reference.broadcast_in_dim(*args, **kwargs)


def symbolic_reshape(*args, **kwargs):
    return lax_reference.reshape(*args, **kwargs)


def symbolic_transpose(*args, **kwargs):
    return lax_reference.transpose(*args, axes=kwargs['permutation'])


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
        result = numpy.full(
            numpy.delete(numpy.shape(operand), axis),
            init_val,
            dtype=numpy.asarray(operand).dtype,
        )

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
        return lax_reference.reduce(
            *args,
            init_value=True,
            computation=numpy.logical_and,
            dimensions=kwargs['axes'],
        )
    else:
        return symbolic_reduce(
            *args,
            init_value='True',
            computation=symbolic_and,
            dimensions=kwargs['axes'],
        )


def symbolic_reduce_or(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.reduce(
            *args,
            init_value=False,
            computation=numpy.logical_or,
            dimensions=kwargs['axes'],
        )
    else:
        return symbolic_reduce(
            *args,
            init_value='False',
            computation=symbolic_or,
            dimensions=kwargs['axes'],
        )


def symbolic_reduce_sum(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.reduce(
            *args, init_value=0, computation=numpy.add, dimensions=kwargs['axes']
        )
    else:
        return symbolic_reduce(
            *args, init_value='0', computation=symbolic_sum, dimensions=kwargs['axes']
        )
