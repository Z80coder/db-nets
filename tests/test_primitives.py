import jax.numpy as jnp
from neurallogic import primitives, neural_logic_net, harden

def test_ravel():
    input = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    expected = jnp.ravel(input)
    output = primitives.ravel(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.ravel(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = jnp.asarray(harden.harden(input)).tolist()
    output = primitives.ravel(neural_logic_net.NetType.Symbolic)(symbolic_input)
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)

def test_ravel_matrix():
    input = jnp.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.88]]])
    expected = jnp.ravel(input)
    output = primitives.ravel(neural_logic_net.NetType.Soft)(input)
    assert jnp.array_equal(output, expected)
    output = primitives.ravel(neural_logic_net.NetType.Hard)(input)
    assert jnp.array_equal(output, expected)
    symbolic_input = jnp.asarray(harden.harden(input)).tolist()
    output = primitives.ravel(neural_logic_net.NetType.Symbolic)(symbolic_input)
    expected = harden.harden(expected)
    assert jnp.array_equal(output, expected)