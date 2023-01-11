import flax
import jax
import numpy


def harden_float(x: float) -> bool:
    return x > 0.5

harden_array = jax.vmap(harden_float, 0, 0)

def harden_list(x: list) -> list:
    return [harden(xi) for xi in x]

def harden_dict(x: dict) -> dict:
    return {k: harden(v) for k, v in x.items()}

# TODO: use dispatch
def harden(x):
    if isinstance(x, float):
        return harden_float(x)
    elif isinstance(x, list):
        return harden_list(x)
    elif isinstance(x, flax.core.frozen_dict.FrozenDict) or isinstance(x, dict):
        return harden_dict(x)
    else:
        # Assuming x is a numpy array
        if x.shape != ():
            return harden_array(x)
        else:
            return numpy.array(harden(x.item()))

def map_keys_nested(f, d: dict) -> dict:
    return {f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()}

def hard_weights(weights):
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Hard"), harden(weights.unfreeze())))

def symbolic_weights(weights):
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Symbolic"), harden(weights.unfreeze())))

