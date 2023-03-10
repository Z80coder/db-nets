import typing
from typing import Any, Mapping

import jax
import numpy
from jax import core
from jax._src.util import safe_map
from plum import dispatch

from neurallogic import symbolic_primitives, map_at_elements

# Imports required for evaluating symbolic expressions with eval()
import jax._src.lax_reference as lax_reference


def symbolic_bind(prim, *args, **params):
    #print('\nprimitive: ', prim.name)
    #print('\targs:\n\t\t', args)
    #print('\tparams\n\t\t: ', params)
    symbolic_outvals = {
        'broadcast_in_dim': symbolic_primitives.symbolic_broadcast_in_dim,
        'reshape': symbolic_primitives.symbolic_reshape,
        'transpose': symbolic_primitives.symbolic_transpose,
        'convert_element_type': symbolic_primitives.symbolic_convert_element_type,
        'eq': symbolic_primitives.symbolic_eq,
        'ne': symbolic_primitives.symbolic_ne,
        'le': symbolic_primitives.symbolic_le,
        'lt': symbolic_primitives.symbolic_lt,
        'ge': symbolic_primitives.symbolic_ge,
        'gt': symbolic_primitives.symbolic_gt,
        'add': symbolic_primitives.symbolic_add,
        'sub': symbolic_primitives.symbolic_sub,
        'mul': symbolic_primitives.symbolic_mul,
        'div': symbolic_primitives.symbolic_div,
        'tan': symbolic_primitives.symbolic_tan,
        'max': symbolic_primitives.symbolic_max,
        'min': symbolic_primitives.symbolic_min,
        'abs': symbolic_primitives.symbolic_abs,
        'round': symbolic_primitives.symbolic_round,
        'floor': symbolic_primitives.symbolic_floor,
        'ceil': symbolic_primitives.symbolic_ceil,
        'and': symbolic_primitives.symbolic_and,
        'or': symbolic_primitives.symbolic_or,
        'xor': symbolic_primitives.symbolic_xor,
        'not': symbolic_primitives.symbolic_not,
        'reduce_and': symbolic_primitives.symbolic_reduce_and,
        'reduce_or': symbolic_primitives.symbolic_reduce_or,
        'reduce_xor': symbolic_primitives.symbolic_reduce_xor,
        'reduce_sum': symbolic_primitives.symbolic_reduce_sum,
        'select_n': symbolic_primitives.symbolic_select_n,
    }[prim.name](*args, **params)
 #   print('\tresult:\n\t\t', symbolic_outvals)
    return symbolic_outvals


def scope_put_variable(self, col: str, name: str, value: Any):
    variables = self._collection(col)

    def put(target, key, val):
        if key in target and isinstance(target[key], dict) and isinstance(val, Mapping):
            for k, v in val.items():
                put(target[key], k, v)
        else:
            target[key] = val

    put(variables, name, value)


def put_variable(self, col: str, name: str, value: Any):
    self.scope._variables = self.scope.variables().unfreeze()
    scope_put_variable(self.scope, col, name, value)


# TODO: make this robust and general over multiple types of param names


def convert_to_numeric_params(flax_layer, param_names: str):
    actual_weights = flax_layer.get_variable('params', param_names)
    # Convert actual weights to dummy numeric weights (if needed)
    if isinstance(actual_weights, list) or (
        isinstance(actual_weights, numpy.ndarray) and actual_weights.dtype == object
    ):
        numeric_weights = map_at_elements.map_at_elements(
            actual_weights, lambda x: 0
        )
        numeric_weights = numpy.asarray(numeric_weights, dtype=numpy.int32)
        put_variable(flax_layer, 'params', param_names, numeric_weights)
    return flax_layer, actual_weights


def make_symbolic_flax_jaxpr(flax_layer, x):
    flax_layer, bit_weights = convert_to_numeric_params(flax_layer, 'bit_weights')
    flax_layer, thresholds = convert_to_numeric_params(flax_layer, 'thresholds')
    # Convert input to dummy numeric input (if needed)
    if isinstance(x, list) or (isinstance(x, numpy.ndarray) and x.dtype == object):
        x = map_at_elements.map_at_elements(x, lambda x: 0)
        x = numpy.asarray(x, dtype=numpy.int32)
    # Make the jaxpr that corresponds to the flax layer
    jaxpr = make_symbolic_jaxpr(flax_layer, x)
    if hasattr(jaxpr, '_consts'):
        # Make a list of bit_weights and thresholds but only include each if they are not None
        bit_weights_and_thresholds = [x for x in [bit_weights, thresholds] if x is not None]
        # Replace the dummy numeric weights with the actual weights in the jaxpr
        jaxpr.__setattr__('_consts', bit_weights_and_thresholds)
    return jaxpr



def eval_jaxpr(symbolic, jaxpr, consts, *args):
    '''Evaluates a jaxpr by interpreting it as Python code.

    Parameters
    ----------
    symbolic : bool
        Whether to return symbolic values or concrete values. If symbolic is
        True, returns symbolic values, and if symbolic is False, returns
        concrete values.
    jaxpr : Jaxpr
        The jaxpr to interpret.
    consts : tuple
        Constant values for the jaxpr.
    args : tuple
        Arguments for the jaxpr.

    Returns
    -------
    out : tuple
        The result of evaluating the jaxpr.
    '''

    # Mapping from variable -> value
    env = {}
    symbolic_env = {}

    # TODO: unify read and symbolic_read

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def symbolic_read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return symbolic_env[var]

    def write(var, val):
        env[var] = val

    def symbolic_write(var, val):
        symbolic_env[var] = val

    # Bind args and consts to environment
    if not symbolic:
        safe_map(write, jaxpr.invars, args)
        safe_map(write, jaxpr.constvars, consts)
    safe_map(symbolic_write, jaxpr.invars, args)
    safe_map(symbolic_write, jaxpr.constvars, consts)

    def eval_jaxpr_impl(jaxpr):
        # Loop through equations and evaluate primitives using `bind`
        for eqn in jaxpr.eqns:
            # Read inputs to equation from environment
            if not symbolic:
                invals = safe_map(read, eqn.invars)
            symbolic_invals = safe_map(symbolic_read, eqn.invars)
            prim = eqn.primitive
            if type(prim) is jax.core.CallPrimitive:
                call_jaxpr = eqn.params['call_jaxpr']
                if not symbolic:
                    safe_map(write, call_jaxpr.invars, map(read, eqn.invars))
                try:
                    safe_map(
                        symbolic_write,
                        call_jaxpr.invars,
                        map(symbolic_read, eqn.invars),
                    )
                except:
                    pass
                eval_jaxpr_impl(call_jaxpr)
                if not symbolic:
                    safe_map(write, eqn.outvars, map(read, call_jaxpr.outvars))
                safe_map(
                    symbolic_write, eqn.outvars, map(symbolic_read, call_jaxpr.outvars)
                )
            else:
                if not symbolic:
                    outvals = prim.bind(*invals, **eqn.params)
                symbolic_outvals = symbolic_bind(prim, *symbolic_invals, **eqn.params)
                # Primitives may return multiple outputs or not
                if not prim.multiple_results:
                    if not symbolic:
                        outvals = [outvals]
                    symbolic_outvals = [symbolic_outvals]
                if not symbolic:
                    # Always check that the symbolic binding generates the same values as the
                    # standard jax binding in order to detect bugs early.
                    # print(f'outvals: {outvals} and symbolic_outvals: {symbolic_outvals}')
                    assert numpy.allclose(
                        numpy.array(outvals), symbolic_outvals, equal_nan=True
                    )
                # Write the results of the primitive into the environment
                if not symbolic:
                    safe_map(write, eqn.outvars, outvals)
                safe_map(symbolic_write, eqn.outvars, symbolic_outvals)

    # Read the final result of the Jaxpr from the environment
    eval_jaxpr_impl(jaxpr)
    if not symbolic:
        return safe_map(read, jaxpr.outvars)[0]
    else:
        return safe_map(symbolic_read, jaxpr.outvars)[0]


def make_symbolic_jaxpr(func: typing.Callable, *args):
    return jax.make_jaxpr(lambda *args: func(*args))(*args)

# TODO: better name
def eval_symbolic(jaxpr, *args):
    if hasattr(jaxpr, 'literals'):
        return eval_jaxpr(
            False, jaxpr.jaxpr, jaxpr.literals, *args
        )
    return eval_jaxpr(False, jaxpr.jaxpr, [], *args)

# TODO: better name
def symbolic_expression(jaxpr, *args):
    if hasattr(jaxpr, 'literals'):
        sym_expr = eval_jaxpr(True, jaxpr.jaxpr, jaxpr.literals, *args)
    else:
        sym_expr = eval_jaxpr(True, jaxpr.jaxpr, [], *args)
    return sym_expr


@dispatch
def eval_symbolic_expression(x: str):
    # TODO: distinguish python code-gen from other possible code-gen
    eval_str = x.replace('inf', 'numpy.inf')
    return eval(eval_str)


@dispatch
def eval_symbolic_expression(x: numpy.ndarray):
    return numpy.vectorize(eval_symbolic_expression)(x)


@dispatch
def eval_symbolic_expression(x: list):
    return numpy.vectorize(eval)(x)

