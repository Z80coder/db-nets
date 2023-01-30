from typing import Callable

import flax
import jax
import numpy
from plum import dispatch

from neurallogic import harden, symbolic_generation, map_at_elements


def to_string(x):
    return str(x)


@dispatch
def make_symbolic(x: dict):
    return map_at_elements.map_at_elements(
        x, to_string
    )


@dispatch
def make_symbolic(x: list):
    return map_at_elements.map_at_elements(
        x, to_string
    )


@dispatch
def make_symbolic(x: numpy.ndarray):
    return map_at_elements.map_at_elements(
        x, to_string
    )


@dispatch
def make_symbolic(x: jax.numpy.ndarray):
    return map_at_elements.map_at_elements(
        convert_jax_to_numpy_arrays(x), to_string
    )


@dispatch
def make_symbolic(x: bool):
    return to_string(x)


@dispatch
def make_symbolic(x: str):
    return to_string(x)


@dispatch
def convert_jax_to_numpy_arrays(x: jax.numpy.ndarray):
    return numpy.asarray(x)


@dispatch
def convert_jax_to_numpy_arrays(x: dict):
    return {k: convert_jax_to_numpy_arrays(v) for k, v in x.items()}


@dispatch
def make_symbolic(x: flax.core.FrozenDict):
    x = convert_jax_to_numpy_arrays(x.unfreeze())
    return flax.core.FrozenDict(make_symbolic(x))


@dispatch
def make_symbolic(*args):
    return tuple([make_symbolic(arg) for arg in args])


def check_consistency(soft: Callable, hard: Callable, expected, *args):
    # print(f'\nchecking consistency for {soft.__name__}')
    # Check that the soft function performs as expected
    soft_output = soft(*args)
    # print(f'Expected: {expected}, Actual soft_output: {repr(soft_output)}')
    assert numpy.allclose(soft_output, expected, equal_nan=True)

    # Check that the hard function performs as expected
    hard_args = tuple([harden.harden(arg) for arg in args])
    hard_expected = harden.harden(expected)
    hard_output = hard(*hard_args)
    # print(f'Expected: {hard_expected}, Actual hard_output: {repr(hard_output)}')
    assert numpy.allclose(hard_output, hard_expected, equal_nan=True)

    # Check that the jaxpr performs as expected
    symbolic_f = symbolic_generation.make_symbolic_jaxpr(hard, *hard_args)
    symbolic_output = symbolic_generation.eval_symbolic(symbolic_f, *hard_args)
    # print(f'Expected: {hard_expected}, Actual symbolic_output: {repr(symbolic_output)}')
    assert numpy.allclose(symbolic_output, hard_expected, equal_nan=True)
