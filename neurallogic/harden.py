import flax
import jax
import numpy
from plum import dispatch
from neurallogic import symbolic_primitives


def harden_float(x: float) -> bool:
    return x > 0.5

"""
harden_array = jax.vmap(harden_float, 0, 0)

def harden_dict(x: dict) -> dict:
    return {k: harden(v) for k, v in x.items()}

@dispatch
def harden(x: float):
    return harden_float(x)

@dispatch
def harden(x: list):
    return [harden(xi) for xi in x]

@dispatch
def harden(x: flax.core.frozen_dict.FrozenDict):
    return harden_dict(x)

@dispatch
def harden(x: dict):
    return harden_dict(x)

@dispatch
def harden(x: numpy.ndarray):
    if x.shape != ():
        return harden_array(x)
    else:
        return numpy.array(harden(x.item()))

@dispatch
def harden(x: jax.numpy.ndarray):
    if x.shape != ():
        return harden_array(x)
    else:
        return numpy.array(harden(x.item()))

@dispatch
def harden(*args):
    #print(f'harden: {args} of type {type(args)} with length {len(args)}')
    #print(f'type of elements are {[type(arg) for arg in args]}')
    #if len(args) == 1:
    #    return harden(args[0])
    return tuple([harden(arg) for arg in args])
"""

@dispatch
def harden(x: float):
    return harden_float(x)

@dispatch
def harden(x: numpy.ndarray):
    return symbolic_primitives.map_at_elements(x, harden_float)

@dispatch
def harden(x: jax.numpy.ndarray):
    return symbolic_primitives.map_at_elements(x, harden_float)

@dispatch
def harden(x: dict):
    return symbolic_primitives.map_at_elements(x, harden_float)

@dispatch
def harden(*args):
    if len(args) == 1:
        return harden(args[0])
    return tuple([harden(arg) for arg in args])

@dispatch
def map_keys_nested(f, d: dict) -> dict:
    return {f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()}


def hard_weights(weights):
    unfrozen_weights = weights.unfreeze()
    hard_weights = harden(unfrozen_weights)
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Hard"), hard_weights))

