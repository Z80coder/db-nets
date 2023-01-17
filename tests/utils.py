from typing import Callable
import numpy
from neurallogic import harden, symbolic_generation

def check_consistency(soft: Callable, hard: Callable, expected, *args):
    # Check that the soft function performs as expected
    assert numpy.allclose(soft(*args), expected)

    # Check that the hard function performs as expected
    hard_args = harden.harden(*args)
    hard_expected = harden.harden(expected)
    assert numpy.allclose(hard(*hard_args), hard_expected)

    # Check that the jaxpr performs as expected
    symbolic_f = symbolic_generation.make_symbolic_jaxpr(hard, *hard_args)
    assert numpy.allclose(symbolic_generation.eval_symbolic(
        symbolic_f, *hard_args), hard_expected)
