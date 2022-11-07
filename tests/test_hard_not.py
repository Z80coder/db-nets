import jax.numpy as jnp
from neurallogic import hard_not

def test_soft_not():
    assert hard_not.soft_not(1.0, 1.0) == 1.0
    assert hard_not.soft_not(1.0, 0.0) == 0.0
    assert hard_not.soft_not(0.0, 0.0) == 1.0
    assert hard_not.soft_not(0.0, 1.0) == 0.0

def test_hard_not():
    assert hard_not.hard_not(True, True) == True
    assert hard_not.hard_not(True, False) == False
    assert hard_not.hard_not(False, False) == True
    assert hard_not.hard_not(False, True) == False

def test_soft_not_neuron():
    x = jnp.array([1.0, 1.0])
    w = jnp.array([1.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([1.0, 1.0]))
    x = jnp.array([0.0, 0.0])
    w = jnp.array([0.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([1.0, 1.0]))
    x = jnp.array([1.0, 0.0])
    w = jnp.array([0.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([0.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([0.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([0.0, 0.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([1.0, 0.0]))
    x = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 1.0])
    assert jnp.array_equal(hard_not.soft_not_neuron(w, x), jnp.array([0.0, 1.0]))

def test_hard_not_neuron():
    x = jnp.array([True, True])
    w = jnp.array([True, True])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([True, True]))
    x = jnp.array([False, False])
    w = jnp.array([False, False])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([True, True]))
    x = jnp.array([True, False])
    w = jnp.array([False, True])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([False, False]))
    x = jnp.array([False, True])
    w = jnp.array([True, False])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([False, False]))
    x = jnp.array([False, True])
    w = jnp.array([False, False])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([True, False]))
    x = jnp.array([False, True])
    w = jnp.array([True, True])
    assert jnp.array_equal(hard_not.hard_not_neuron(w, x), jnp.array([False, True]))

def test_soft_not_layer():
    x = jnp.array([1.0, 0.0])
    w = jnp.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    expected_value = jnp.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    result = hard_not.soft_not_layer(w, x)
    assert jnp.array_equal(result, expected_value)

def test_hard_not_layer():
    x = jnp.array([True, False])
    w = jnp.array([[True, True], [False, True], [True, False], [False, False]])
    expected_value = jnp.array([[True, False], [False, False], [True, True], [False, True]])
    result = hard_not.hard_not_layer(w, x)
    assert jnp.array_equal(result, expected_value)
