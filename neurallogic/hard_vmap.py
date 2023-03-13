import jax
import numpy

from neurallogic import neural_logic_net


def soft_vmap(f):
    return jax.vmap(f)


def hard_vmap(f):
    return soft_vmap(f)


def symbolic_vmap(f):
    return numpy.vectorize(f, otypes=[object])
    
vmap = neural_logic_net.select(
    lambda f: soft_vmap(f[0]),
    lambda f: hard_vmap(f[1]),
    lambda f: symbolic_vmap(f[2])
)

# TODO: add tests
