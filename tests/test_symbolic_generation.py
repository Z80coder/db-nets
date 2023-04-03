import flax
import jax
import jax.numpy as jnp
import numpy

from neurallogic import (hard_and, hard_majority, hard_not, hard_or, hard_xor,
                         harden, harden_layer, neural_logic_net, real_encoder,
                         symbolic_generation, hard_concatenate, hard_vmap, symbolic_primitives)
from tests import utils


def nln(type, x, width):
    # Can't symbolically support this layer yet since the symbolic output is an unevaluated string that
    # lacks the correct tensor structure
    # x = real_encoder.real_encoder_layer(type)(2)(x)
    # x = x.ravel()
    y = hard_vmap.vmap(type)((lambda x: 1 - x, lambda x: 1 - x, lambda x: symbolic_primitives.symbolic_not(x)))(x)
    x = hard_concatenate.concatenate(type)([x, y], 0)
    x = hard_or.or_layer(type)(width)(x)
    x = hard_and.and_layer(type)(width)(x)
    x = hard_xor.xor_layer(type)(width)(x)
    x = hard_not.not_layer(type)(2)(x)
    x = hard_majority.majority_layer(type)()(x)
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape([2, 1])
    x = x.sum(-1)
    return x


def test_symbolic_generation():
    # Define width of network
    width = 2
    # Define the neural logic net
    soft, hard, symbolic = neural_logic_net.net(
        lambda type, x: nln(type, x, width))
    # Initialize a random number generator
    rng = jax.random.PRNGKey(0)
    #rng, init_rng = jax.random.split(rng)
    mock_input = harden.harden(jnp.ones([2 * 2]))
    # Initialize the weights of the neural logic net
    soft_weights = soft.init(rng, mock_input)
    hard_weights = harden.hard_weights(soft_weights)
    # Apply the neural logic net to the hard input
    hard_output = hard.apply(hard_weights, mock_input)

    # Check the standard evaluation of the network equals the non-standard evaluation
    symbolic_weights = harden.hard_weights(soft_weights)
    symbolic_output = symbolic.apply(symbolic_weights, mock_input)
    assert numpy.array_equal(symbolic_output, hard_output)

    # Check the standard evaluation of the network equals the non-standard symbolic evaluation
    symbolic_mock_input = utils.make_symbolic(mock_input)
    symbolic_output = symbolic.apply(symbolic_weights, symbolic_mock_input)
    assert numpy.array_equal(hard_output.shape, symbolic_output.shape)

    # Compute the symbolic expression, i.e. perform the actual operations in the symbolic expression
    eval_symbolic_output = symbolic_generation.eval_symbolic_expression(symbolic_output)
    # If this assertion succeeds then the non-standard symbolic evaluation of the jaxpr is is identical to the standard evaluation of network
    assert numpy.array_equal(hard_output, eval_symbolic_output)
