import jax
import flax

def harden(x: float) -> bool:
    return x > 0.5

def harden_list(x: list) -> list:
    return [harden(xi) for xi in x]

harden_array = jax.vmap(harden, 0, 0)

def harden_item(x):
    if isinstance(x, flax.core.frozen_dict.FrozenDict):
        return harden_dict(x)
    else:
        return harden_array(x)

def harden_dict(x: dict) -> dict:
    return {k: harden_item(v) for k, v in x.items()}

