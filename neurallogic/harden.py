import jax

def harden(x: float) -> bool:
    return x > 0.5

def harden_list(x: list) -> list:
    return [harden(xi) for xi in x]

harden_array = jax.vmap(harden, 0, 0)
