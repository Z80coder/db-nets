import jax
import jax._src.lax_reference as lax_reference
from jax import core
from jax._src.util import safe_map
import numpy
from neurallogic import symbolic_primitives, harden


def symbolic_bind(prim, *args, **params):
    # print("\n---symbolic_bind:")
    # print("primitive: ", prim.name)
    # print("args: ", args)
    # print("params: ", params)
    symbolic_outvals = {
        'broadcast_in_dim': symbolic_primitives.symbolic_broadcast_in_dim,
        'reshape': lax_reference.reshape,
        'convert_element_type': symbolic_primitives.symbolic_convert_element_type,
        'and': symbolic_primitives.symbolic_and,
        'or': symbolic_primitives.symbolic_or,
        'xor': symbolic_primitives.symbolic_xor,
        'not': symbolic_primitives.symbolic_not,
        'reduce_or': symbolic_primitives.symbolic_reduce_or,
        'reduce_sum': symbolic_primitives.symbolic_reduce_sum,
    }[prim.name](*args, **params)
    return symbolic_outvals


def eval_jaxpr(symbolic, jaxpr, consts, *args):
    """Evaluates a jaxpr by interpreting it as Python code.

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
    """

    # Mapping from variable -> value
    env = {}
    symbolic_env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def symbolic_read(var):
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
                    safe_map(symbolic_write, call_jaxpr.invars,
                             map(symbolic_read, eqn.invars))
                except:
                    pass
                eval_jaxpr_impl(call_jaxpr)
                if not symbolic:
                    safe_map(write, eqn.outvars, map(read, call_jaxpr.outvars))
                safe_map(symbolic_write, eqn.outvars, map(
                    symbolic_read, call_jaxpr.outvars))
            else:
                if not symbolic:
                    outvals = prim.bind(*invals, **eqn.params)
                symbolic_outvals = symbolic_bind(
                    prim, *symbolic_invals, **eqn.params)
                # Primitives may return multiple outputs or not
                if not prim.multiple_results:
                    if not symbolic:
                        outvals = [outvals]
                    symbolic_outvals = [symbolic_outvals]
                if not symbolic:
                    # print(f"outvals: {type(outvals)}: {outvals}")
                    # print(
                    #    f"symbolic_outvals: {type(symbolic_outvals)}: {symbolic_outvals}")
                    # Check that the concrete and symbolic values are equal
                    assert numpy.array_equal(
                        numpy.array(outvals), symbolic_outvals)
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


def make_symbolic_net(net, net_weights, mock_net_input):
    return jax.make_jaxpr(lambda x: net.apply(net_weights, x))(harden.harden(mock_net_input))


def eval_symbolic_net(jaxpr, *args):
    return eval_jaxpr(False, jaxpr.jaxpr, jaxpr.literals, *args)


def compute_symbolic_output(symbolic_net, *args):
    symbolic_jaxpr_literals = safe_map(
        lambda x: numpy.array(x, dtype=object), symbolic_net.literals)
    symbolic_jaxpr_literals = symbolic_primitives.to_boolean_symbolic_values(
        symbolic_jaxpr_literals)
    return eval_jaxpr(True, symbolic_net.jaxpr, symbolic_jaxpr_literals, *args)


