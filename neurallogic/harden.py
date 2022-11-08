import jax
import flax

def harden_float(x: float) -> bool:
    return x > 0.5

harden_array = jax.vmap(harden_float, 0, 0)

def harden_list(x: list) -> list:
    return [harden(xi) for xi in x]

def harden_dict(x: dict) -> dict:
    return {k: harden(v) for k, v in x.items()}

def harden(x):
    if isinstance(x, float):
        return harden_float(x)
    elif isinstance(x, list):
        return harden_list(x)
    elif isinstance(x, flax.core.frozen_dict.FrozenDict) or isinstance(x, dict):
        return harden_dict(x)
    else:
        return harden_array(x)

