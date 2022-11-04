import jax.numpy as jnp
from neurallogic import hard_not

def test_soft_not():
    assert hard_not.soft_not(1.0, 0.0) == 0.0
    assert hard_not.soft_not(1.0, 1.0) == 1.0
    assert hard_not.soft_not(1.0, 1.0) == 1.0
    assert hard_not.soft_not(0.0, 0.0) == 1.0
    assert hard_not.soft_not(0.0, 1.0) == 0.0

def test_soft_not_neuron():
    x = jnp.array([1.0, 1.0])
    w = jnp.array([1.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([1.0, 1.0]))
    x = jnp.array([0.0, 0.0])
    w = jnp.array([0.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([1.0, 1.0]))
    x = jnp.array([1.0, 0.0])
    w = jnp.array([0.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([0.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([0.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([0.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([1.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not(w, x), jnp.array([0.0, 1.0]))

def test_soft_not_layer():
    x = jnp.array([1.0, 0.0])
    w = jnp.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    expected_value = jnp.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    result = hard_not.soft_not_layer(w, x)
    assert jnp.array_equal(result, expected_value)
