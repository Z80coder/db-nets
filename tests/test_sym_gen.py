from neurallogic import neural_logic_net, harden, hard_or, sym_gen, symbolic_primitives
from tests import test_mnist
import numpy
import jax
import jax.numpy as jnp
from flax import linen as nn

def nln(type, x, width):
    x = hard_or.or_layer(type)(width, nn.initializers.uniform(1.0), dtype=jnp.float32)(x) 
    #x = hard_not.not_layer(type)(10, dtype=jnp.float32)(x)
    #x = primitives.nl_ravel(type)(x) 
    #x = harden_layer.harden_layer(type)(x) 
    #x = primitives.nl_reshape(type)((10, width))(x) 
    #x = primitives.nl_sum(type)(-1)(x) 
    return x

def batch_nln(type, x, width):
    return jax.vmap(lambda x: nln(type, x, width))(x)

def test_sym_gen():
    train_ds, test_ds = test_mnist.get_datasets()
    train_ds["image"] = jnp.reshape(train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(test_ds["image"], (test_ds["image"].shape[0], -1))

    width = 10
    soft, hard, _ = neural_logic_net.net(lambda type, x: nln(type, x, width))

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    mock_input = harden.harden(jnp.ones([28 * 28]))
    hard_weights = harden.hard_weights(soft.init(rng, mock_input))

    jaxpr = jax.make_jaxpr(lambda x: hard.apply(hard_weights, x))(harden.harden(test_ds['image'][0]))

    hard_mock_input = harden.harden(test_ds['image'][0])
    hard_output = hard.apply(hard_weights, hard_mock_input)
    #print("hard_output shape:", hard_output.shape)
    #print("hard_output:", hard_output)
    eval_hard_output, eval_symbolic_output = sym_gen.eval_jaxpr(False, jaxpr.jaxpr, jaxpr.literals, hard_mock_input)
    #print("eval_hard_output:", eval_hard_output)
    #print("eval_symbolic_output:", eval_symbolic_output)
    assert numpy.array_equal(numpy.array(eval_hard_output), eval_symbolic_output)
    print("SUCCESS: jax primitives and symbolic primitives are identical.")
    standard_jax_output = hard.apply(hard_weights, hard_mock_input)
    #print("standard_jax_output", standard_jax_output)
    #print("eval_hard_output", numpy.array(eval_hard_output))
    assert jax.numpy.array_equal(numpy.array(eval_hard_output), standard_jax_output)
    print("SUCCESS: non-standard evaluation is identical to standard evaluation of jaxpr.")
    symbolic_mock_input = symbolic_primitives.to_boolean_string(hard_mock_input)
    #print("symbolic_mock_input:", symbolic_mock_input)
    #print("type of symbolic_mock_input = ", type(symbolic_mock_input))
    #print("type of element = ", symbolic_mock_input.dtype)
    symbolic_jaxpr_literals = symbolic_primitives.to_boolean_string(jaxpr.literals)
    #print("jaxpr.literals = ", symbolic_jaxpr_literals)
    #print("type of jaxpr.literals = ", type(symbolic_jaxpr_literals))
    #print("type of element = ", symbolic_jaxpr_literals.dtype)
    eval_symbolic_output = sym_gen.eval_jaxpr(True, jaxpr.jaxpr, symbolic_jaxpr_literals, symbolic_mock_input)
    # assert the dimensions of eval_hard_output and eval_symbolic_output are the same
    eval_hard_output = numpy.array(eval_hard_output)
    print("eval_hard_output", eval_hard_output)
    #print("type of eval_hard_output = ", type(eval_hard_output))
    #print("eval_symbolic_output:", eval_symbolic_output)
    #print("type of eval_symbolic_output = ", type(eval_symbolic_output))
    #print("shape of eval_hard_output = ", eval_hard_output.shape)
    #print("shape of eval_symbolic_output = ", eval_symbolic_output.shape)
    assert numpy.array_equal(eval_hard_output.shape, eval_symbolic_output.shape)
    print("SUCCESS: dimensions of non-standard evaluation and standard evaluation of jaxpr are identical.")
    # assert the values of eval_hard_output and eval_symbolic_output are the same
    reduced_eval_symbolic_output = symbolic_primitives.symbolic_eval(eval_symbolic_output)
    print("reduced_eval_symbolic_output:", reduced_eval_symbolic_output)
    assert numpy.array_equal(eval_hard_output, reduced_eval_symbolic_output)
    print("SUCCESS: values of symbolic evaluation and standard evaluation of jaxpr are identical.")