import pytest
import jax
import jax.numpy as jnp
from neurallogic import hard_not

def test_differentiable_hard_not():
    x = jnp.array([1.0])
    w = jnp.array([0.0])
    assert hard_not.differentiable_hard_not(x, w) == pytest.approx([0.0])
    x = jnp.array([1.0])
    w = jnp.array([1.0])
    assert hard_not.differentiable_hard_not(x, w) == pytest.approx([1.0])
    x = jnp.array([0.0])
    w = jnp.array([0.0])
    assert hard_not.differentiable_hard_not(x, w) == pytest.approx([1.0])
    x = jnp.array([0.0])
    w = jnp.array([1.0])
    assert hard_not.differentiable_hard_not(x, w) == pytest.approx([0.0])
    x = jnp.array([1.0, 0.0, 1.0, 0.0])
    w = jnp.array([1.0, 1.0, 0.0, 0.0])
    assert hard_not.differentiable_hard_not(x, w) == pytest.approx([1.0, 0.0, 0.0, 1.0])

def test_threaded_differentiable_hard_not():
    threaded_differentiable_hard_not = jax.vmap(hard_not.differentiable_hard_not, in_axes=1, out_axes=0)
    x = jnp.array([1.0, 0.0, 1.0, 0.0])
    w = jnp.array([1.0, 1.0, 0.0, 0.0])
    xs = jnp.stack([x for _ in range(10)])
    ws = jnp.stack([w for _ in range(10)])
    result = threaded_differentiable_hard_not(xs, ws)
    expected_result = jnp.stack([jnp.array([1.0, 0.0, 0.0, 1.0]) for _ in range(10)])
    print("x = ", xs)
    print("w = ", ws)
    print("result = ", result)
    print("expected result = ", expected_result)
    assert jnp.array_equal(result, expected_result)
