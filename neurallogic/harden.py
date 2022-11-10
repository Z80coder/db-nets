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

def map_keys_nested(f, d: dict) -> dict:
    return {f(k): map_keys_nested(f, v) if isinstance(v, dict) else v for k, v in d.items()}

def harden_weights(weights):
    return flax.core.FrozenDict(map_keys_nested(lambda str: str.replace("Soft", "Hard"), harden(weights.unfreeze())))
