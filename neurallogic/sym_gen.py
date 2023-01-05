import jax
import jax._src.lax_reference as lax_reference
from jax import core
from jax._src.util import safe_map
import numpy
from neurallogic import symbolic_primitives

def symbolic_broadcast_in_dim(*args, **kwargs):
  #print("broadcasting type", args[0].dtype)
  r = lax_reference.broadcast_in_dim(*args, **kwargs)
  #print("broadcasting result type", r.dtype)
  return r

def make_symbolic_reducer(py_binop, init_val):
  def reducer(operand, axis=0):
    axis = range(numpy.ndim(operand)) if axis is None else axis
    result = numpy.full(numpy.delete(numpy.shape(operand), axis), init_val, dtype=numpy.asarray(operand).dtype)
    for idx, _ in numpy.ndenumerate(operand):
      out_idx = tuple(numpy.delete(idx, axis))
      result[out_idx] = py_binop(result[out_idx], operand[idx])
    return result
  return reducer

def symbolic_reduce(operand, init_value, computation, dimensions):
  reducer = make_symbolic_reducer(computation, init_value)
  return reducer(operand, tuple(dimensions)).astype(operand.dtype)
  
def symbolic_reduce_or(*args, **kwargs):
  if args[0].dtype == bool:
    return lax_reference.reduce(*args, init_value=False, dimensions=kwargs['axes'], computation=numpy.logical_or)
  else:
    #print("args = ", args)
    #print("type args[0] = ", type(args[0]))
    #print("element type = ", args[0].dtype)
    r = symbolic_reduce(*args, init_value='False', dimensions=kwargs['axes'], computation=symbolic_primitives.symbolic_or)
    #print("symbolic_reduce_or result: ", r)
    #print("type: ", type(r))
    #print("element type: ", r.dtype)
    return r

def symbolic_bind(prim, *args, **params):
  #print("primitive: ", prim.name)
  symbolic_outvals = {
    'and': symbolic_primitives.symbolic_and,
    'broadcast_in_dim': symbolic_broadcast_in_dim,
    'xor': symbolic_primitives.symbolic_xor,
    'not': symbolic_primitives.symbolic_not,
    'reshape': lax_reference.reshape,
    'reduce_or': symbolic_reduce_or,
  }[prim.name](*args, **params)
  return symbolic_outvals

def eval_jaxpr(symbolic, jaxpr, consts, *args):
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
      # `bind` is how a primitive is called
      prim = eqn.primitive
      if type(prim) is jax.core.CallPrimitive:
        call_jaxpr = eqn.params['call_jaxpr']
        if not symbolic:
          safe_map(write, call_jaxpr.invars, map(read, eqn.invars))
        safe_map(symbolic_write, call_jaxpr.invars, map(symbolic_read, eqn.invars))
        eval_jaxpr_impl(call_jaxpr)
        if not symbolic:
          safe_map(write, eqn.outvars, map(read, call_jaxpr.outvars))
        safe_map(symbolic_write, eqn.outvars, map(symbolic_read, call_jaxpr.outvars))
      else:
        if not symbolic:
          outvals = prim.bind(*invals, **eqn.params)
        symbolic_outvals = symbolic_bind(prim, *symbolic_invals, **eqn.params)
        #if not symbolic:
        #  print(f"outvals: {type(outvals)}: {outvals.shape}: {outvals}")
        #print(f"symbolic_outvals: {type(symbolic_outvals)}: {symbolic_outvals.shape}: {symbolic_outvals}")
        # Primitives may return multiple outputs or not
        if not prim.multiple_results: 
          if not symbolic:
            outvals = [outvals]
          symbolic_outvals = [symbolic_outvals]
        if not symbolic:
          assert numpy.array_equal(numpy.array(outvals), symbolic_outvals)
        # Write the results of the primitive into the environment
        if not symbolic:
          safe_map(write, eqn.outvars, outvals)
        safe_map(symbolic_write, eqn.outvars, symbolic_outvals)

  # Read the final result of the Jaxpr from the environment
  eval_jaxpr_impl(jaxpr)
  if not symbolic:
    val, symbolic_val = safe_map(read, jaxpr.outvars), safe_map(symbolic_read, jaxpr.outvars)
    return val[0], symbolic_val[0]
  else:
    return safe_map(symbolic_read, jaxpr.outvars)[0]