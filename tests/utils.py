from typing import Callable
import numpy
from neurallogic import harden, symbolic_generation

def check_consistency(soft: Callable, hard: Callable, expected, *args):
    #print(f'\nchecking consistency for {soft.__name__}')
    # Check that the soft function performs as expected
    soft_output = soft(*args)
    #print(f'Expected: {expected}, Actual soft_output: {soft_output}')
    assert numpy.allclose(soft_output, expected, equal_nan=True)

    # Check that the hard function performs as expected
    hard_args = harden.harden(*args)
    hard_expected = harden.harden(expected)
    hard_output = hard(*hard_args) 
    #print(f'Expected: {hard_expected}, Actual hard_output: {hard_output}')
    assert numpy.allclose(hard_output, hard_expected, equal_nan=True)

    # Check that the jaxpr performs as expected
    symbolic_f = symbolic_generation.make_symbolic_jaxpr(hard, *hard_args)
    symbolic_output = symbolic_generation.eval_symbolic(symbolic_f, *hard_args)
    #print(f'Expected: {hard_expected}, Actual symbolic_output: {symbolic_output}')
    assert numpy.allclose(symbolic_output, hard_expected, equal_nan=True)
