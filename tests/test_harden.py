import jax.numpy as jnp
from neurallogic import harden

def test_harden():
    assert harden.harden(0.5) == False
    assert harden.harden(0.6) == True
    assert harden.harden(0.4) == False
    assert harden.harden(0.0) == False
    assert harden.harden(1.0) == True

def test_harden_list():
    assert harden.harden_list([0.5, 0.6, 0.4, 0.0, 1.0]) == [False, True, False, False, True]

def test_harden_array():
    assert jnp.array_equal(harden.harden_array(jnp.array([0.5, 0.6, 0.4, 0.0, 1.0])), [False, True, False, False, True])