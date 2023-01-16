import flax
import jax
import numpy
from plum import dispatch


def harden_float(x: float) -> bool:
    return x > 0.5

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


def map_keys_nested(f, d: dict) -> dict:
    return {f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()}

def hard_weights(weights):
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Hard"), harden(weights.unfreeze())))

