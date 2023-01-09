from neurallogic import neural_logic_net, harden, hard_or, hard_not, sym_gen, symbolic_primitives
from tests import test_mnist
import numpy
import jax
import jax.numpy as jnp
from flax import linen as nn


def nln(type, x, width):
    x = hard_or.or_layer(type)(width)(x)
    x = hard_not.not_layer(type)(2)(x)
    # x = primitives.nl_ravel(type)(x)
    # x = harden_layer.harden_layer(type)(x)
    # x = primitives.nl_reshape(type)((10, width))(x)
    # x = primitives.nl_sum(type)(-1)(x)
    return x


def batch_nln(type, x, width):
    return jax.vmap(lambda x: nln(type, x, width))(x)


def test_sym_gen():
    # Get MNIST dataset
    train_ds, test_ds = test_mnist.get_datasets()
    # Flatten images
    train_ds["image"] = jnp.reshape(
        train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(
        test_ds["image"], (test_ds["image"].shape[0], -1))
    # Define width of network
    width = 10
    # Define the neural logic net
    soft, hard, _ = neural_logic_net.net(lambda type, x: nln(type, x, width))
    # Initialize a random number generator
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    mock_input = harden.harden(jnp.ones([28 * 28]))
    # Initialize the weights of the neural logic net
    hard_weights = harden.hard_weights(soft.init(rng, mock_input))
    # Create a jaxpr from the neural logic net (with an arbitrary image input to set sizes)
    jaxpr = jax.make_jaxpr(lambda x: hard.apply(hard_weights, x))(
        harden.harden(test_ds['image'][0]))
    # Define a hard mock input
    hard_mock_input = harden.harden(test_ds['image'][0])
    # Apply the neural logic net to the hard input (not used for testing, just to check that it works)
    hard_output = hard.apply(hard_weights, hard_mock_input)
    # Symbolically evaluate the jaxpr with the hard input
    eval_hard_output, eval_symbolic_output = sym_gen.eval_jaxpr(
        False, jaxpr.jaxpr, jaxpr.literals, hard_mock_input)
    # Check that the hard output and the symbolic output are identical
    assert numpy.array_equal(eval_hard_output, eval_symbolic_output)
    # Evaluate the jaxpr with the hard input using standard jax evaluation
    standard_jax_output = hard.apply(hard_weights, hard_mock_input)
    # Check that the hard output and the standard jax output are identical
    # If this assertion succeeds then the non-standard evaluation of the jaxpr is is identical to the standard evaluation of the jaxpr
    assert jax.numpy.array_equal(numpy.array(
        eval_hard_output), standard_jax_output)
    # Convert the hard input to a symbolic input
    symbolic_mock_input = symbolic_primitives.to_boolean_symbolic_values(
        hard_mock_input)
    # Convert the literals to symbolic literals
    symbolic_jaxpr_literals = symbolic_primitives.to_boolean_symbolic_values(
        jaxpr.literals)
    # Symbolically evaluate the jaxpr with the symbolic input
    eval_symbolic_output = sym_gen.eval_jaxpr(
        True, jaxpr.jaxpr, symbolic_jaxpr_literals, symbolic_mock_input)
    # Check that the hard output and the symbolic output are identical
    eval_hard_output = numpy.array(eval_hard_output)
    # If this assertion succeeds then the shape of the non-standard symbolic evaluation is identical to the shape of the standard evaluation of the jaxpr
    assert numpy.array_equal(eval_hard_output.shape,
                             eval_symbolic_output.shape)
    # Now we check that their values are the same
    # Reduce the symbolic output to a python boolean array
    reduced_eval_symbolic_output = symbolic_primitives.symbolic_eval(
        eval_symbolic_output)
    # Check that the hard output and the reduced symbolic output are identical
    assert numpy.array_equal(eval_hard_output, reduced_eval_symbolic_output)
