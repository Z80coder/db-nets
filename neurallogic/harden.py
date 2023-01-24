import flax
import jax
import numpy
from plum import dispatch

from neurallogic import map_at_elements


def harden_float(x: float) -> bool:
    return x > 0.5


harden_array = jax.vmap(harden_float, 0, 0)


@dispatch
def harden(x: float):
    if numpy.isnan(x):
        return x
    return harden_float(x)


@dispatch
def harden(x: bool):
    return x


@dispatch
def harden(x: list):
    return map_at_elements.map_at_elements(x, harden_float)


@dispatch
def harden(x: numpy.ndarray):
    if x.ndim == 0:
        return harden(x.item())
    return harden_array(x)


@dispatch
def harden(x: jax.numpy.ndarray):
    if x.ndim == 0:
        return harden(x.item())
    return harden_array(x)


@dispatch
def harden(x: dict):
    # Only harden parameters that explicitly represent bits
    def conditional_harden(k, v):
        if k.startswith("bit_"):
            return map_at_elements.map_at_elements(v, harden)
        elif isinstance(v, dict) or isinstance(v, flax.core.FrozenDict) or isinstance(v, list):
            return harden(v)
        return v

    return {k: conditional_harden(k, v) for k, v in x.items()}


@dispatch
def harden(x: flax.core.FrozenDict):
    return harden(x.unfreeze())


"""
@dispatch
def harden(*args):
    if len(args) == 1:
        print(f'args = {args} of type {type(args)}')
        arg = args[0]
        print(f'args[0] = {arg}')
        return tuple(harden(arg))
    return tuple([harden(arg) for arg in args])
"""

@dispatch
def map_keys_nested(f, d: dict) -> dict:
    return {
        f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()
    }


def hard_weights(weights):
    return flax.core.FrozenDict(
        map_keys_nested(
            lambda str: str.replace("Soft", "Hard"), harden(weights.unfreeze())
        )
    )
