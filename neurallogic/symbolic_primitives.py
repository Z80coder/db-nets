import jax
import jax._src.lax_reference as lax_reference
import numpy

from neurallogic import symbolic_operator, symbolic_representation


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
        return symbolic_operator.symbolic_operator('numpy.logical_not', *args, **kwargs)


def symbolic_eq(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.eq(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.eq', *args, **kwargs)


def symbolic_ne(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.ne(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.ne', *args, **kwargs)


def symbolic_le(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.le(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.le', *args, **kwargs)


def symbolic_lt(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.lt(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.lt', *args, **kwargs)


def symbolic_ge(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.ge(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.ge', *args, **kwargs)


def symbolic_gt(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.gt(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.gt', *args, **kwargs)


def symbolic_abs(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.abs(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.absolute', *args, **kwargs)


def symbolic_add(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.add(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.add', *args, **kwargs)


def symbolic_sub(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.sub(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.subtract', *args, **kwargs)


def symbolic_mul(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.mul(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.multiply', *args, **kwargs)


def symbolic_div(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.div(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.div', *args, **kwargs)


def symbolic_max(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.max(*args, **kwargs)
    else:
        r = symbolic_operator.symbolic_operator('numpy.maximum', *args, **kwargs)
        return r


def symbolic_min(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.min(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.minimum', *args, **kwargs)


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
        # TODO: need a more general solution to unquoting symbolic strings
        evaluable_pred = symbolic_representation.symbolic_representation(pred)
        evaluable_on_true = symbolic_representation.symbolic_representation(on_true)
        evaluable_on_false = symbolic_representation.symbolic_representation(on_false)
        return f'lax_reference.select({evaluable_pred}, {evaluable_on_false}, {evaluable_on_true})'


def symbolic_and(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_and(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.logical_and', *args, **kwargs)


def symbolic_or(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_or(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.logical_or', *args, **kwargs)


def symbolic_xor(*args, **kwargs):
    if all_concrete_values([*args]):
        return numpy.logical_xor(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('numpy.logical_xor', *args, **kwargs)


def symbolic_sum(*args, **kwargs):
    if all_concrete_values([*args]):
        return lax_reference.sum(*args, **kwargs)
    else:
        return symbolic_operator.symbolic_operator('lax_reference.sum', *args, **kwargs)


def symbolic_broadcast_in_dim(*args, **kwargs):
    # broadcast_in_dim requires numpy arrays not lists
    args = tuple([numpy.array(arg) if isinstance(
        arg, list) else arg for arg in args])
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
        # Otherwise, we nop
        def convert_element_type(x, dtype):
            return x
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
