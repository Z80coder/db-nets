import jax
import jax.numpy as jnp
import numpy

from neurallogic import (hard_not, hard_or, harden, harden_layer,
                         neural_logic_net, symbolic_generation,
                         symbolic_primitives)
from tests import test_mnist, utils


def nln(type, x, width):
    x = hard_or.or_layer(type)(width)(x)
    x = hard_not.not_layer(type)(10)(x)
    x = x.ravel()
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((10, width))
    x = x.sum(-1)
    return x


def test_symbolic_generation():
    # Get MNIST dataset
    train_ds, test_ds = test_mnist.get_datasets()
    # Flatten images
    train_ds["image"] = jnp.reshape(
        train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(
        test_ds["image"], (test_ds["image"].shape[0], -1))

    # Define width of network
    width = 2
    # Define the neural logic net
    soft, hard, symbolic = neural_logic_net.net(
        lambda type, x: nln(type, x, width))
    # Initialize a random number generator
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    mock_input = harden.harden(jnp.ones([28 * 28]))
    # Initialize the weights of the neural logic net
    hard_weights = harden.hard_weights(soft.init(rng, mock_input))
    # Define a hard mock input
    hard_mock_input = harden.harden(test_ds['image'][0])
    # Apply the neural logic net to the hard input
    hard_output = hard.apply(hard_weights, hard_mock_input)

    # Check the standard evaluation of the network equals the non-standard evaluation
    symbolic_output = symbolic.apply(hard_weights, hard_mock_input)
    assert numpy.array_equal(symbolic_output, hard_output)

    # Check the standard evaluation of the network equals the non-standard symbolic evaluation
    symbolic_mock_input = utils.make_symbolic(hard_mock_input)
    symbolic_output = symbolic.apply(hard_weights, symbolic_mock_input)
    assert numpy.array_equal(hard_output.shape, symbolic_output.shape)

    # Compute the symbolic expression, i.e. perform the actual operations in the symbolic expression
    #print(f'symbolic_output: {symbolic_output}')
    # TODO: We cannot evaluate the symbolic expression because it has too many nested parantheses
    #eval_symbolic_output = symbolic_generation.eval_symbolic_expression(symbolic_output)
    # If this assertion succeeds then the non-standard symbolic evaluation of the jaxpr is is identical to the standard evaluation of network
    #assert numpy.array_equal(hard_output, eval_symbolic_output)
