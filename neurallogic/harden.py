import flax
import jax
import numpy
from plum import dispatch
from neurallogic import symbolic_primitives


def harden_float(x: float) -> bool:
    return x > 0.5


harden_array = jax.vmap(harden_float, 0, 0)

@dispatch
def harden(x: float):
    if numpy.isnan(x):
        return x
    return harden_float(x)

@dispatch
def harden(x: list):
    return symbolic_primitives.map_at_elements(x, harden_float)

@dispatch
def harden(x: numpy.ndarray):
    return harden_array(x)

@dispatch
def harden(x: jax.numpy.ndarray):
    return harden_array(x)

@dispatch
def harden(x: dict):
    return symbolic_primitives.map_at_elements(x, harden_float)

@dispatch
def harden(x: flax.core.FrozenDict):
    return flax.core.FrozenDict(symbolic_primitives.map_at_elements(x.unfreeze(), harden_float))

@dispatch
def harden(*args):
    if len(args) == 1:
        return harden(args[0])
    return tuple([harden(arg) for arg in args])

@dispatch
def map_keys_nested(f, d: dict) -> dict:
    return {f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()}

def hard_weights(weights):
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Hard"), harden(weights.unfreeze())))
