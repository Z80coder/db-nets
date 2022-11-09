import jax
import jax.numpy as jnp
from jax import random
from neurallogic import hard_not
from neurallogic import harden
import optax
from flax.training import train_state

def test_activation():
    test_data = [
        # Test logic
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 0.0],
        # test clipping
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 0.0]
    ]
    for input, expected in test_data:
        assert hard_not.soft_not(*input) == expected
        assert hard_not.hard_not(*harden.harden_list(input)) == harden.harden(expected)
        assert eval(hard_not.symbolic_not(*harden.harden_list(input))) == harden.harden(expected)

def test_neuron():
    test_data = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    ]
    for input, weights, expected in test_data:
        assert jnp.array_equal(hard_not.soft_not_neuron(jnp.array(weights), jnp.array(input)), jnp.array(expected))
        assert jnp.array_equal(hard_not.hard_not_neuron(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input))), harden.harden_array(jnp.array(expected)))
        assert jnp.array_equal(eval(hard_not.symbolic_not_neuron(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input)))), harden.harden_array(jnp.array(expected)))

def test_layer():
    test_data = [
        [[1.0, 0.0], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2]], [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.8]]],
        [[1.0, 0.4], [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 0.4], [0.0, 0.4], [1.0, 0.6], [0.0, 0.6]]],
        [[0.0, 1.0], [[1.0, 1.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.8], [0.0, 0.0], [1.0, 0.0]]],
        [[0.0, 0.0], [[1.0, 0.01], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 0.99], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    ]
    for input, weights, expected in test_data:
        assert jnp.array_equal(hard_not.soft_not_layer(jnp.array(weights), jnp.array(input)), jnp.array(expected))
        assert jnp.array_equal(hard_not.hard_not_layer(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input))), harden.harden_array(jnp.array(expected)))
        assert jnp.array_equal(eval(hard_not.symbolic_not_layer(harden.harden_array(jnp.array(weights)), harden.harden_array(jnp.array(input)))), harden.harden_array(jnp.array(expected)))

def test_not():
    soft_net, hard_net, symbolic_net = hard_not.NotLayer(layer_size=4)
    soft_weights = soft_net.init(random.PRNGKey(0), [1.0, 1.0])
    hard_weights = harden.harden_dict(soft_weights)
    test_data = [
        [
            [1.0, 1.0],
            [[0.26377404, 0.8707025], [0.44444847, 0.03216302], [0.6110164, 0.685097], [0.9133855, 0.08662593]]
        ],
        [
            [1.0, 0.0],
            [[0.26377404, 0.1292975], [0.44444847, 0.967837], [0.6110164, 0.31490302], [0.9133855, 0.91337407]]
        ],
        [
            [0.0, 1.0],
            [[0.73622596, 0.8707025], [0.5555515, 0.03216302], [0.3889836, 0.685097], [0.08661449, 0.08662593]]
        ],
        [
            [0.0, 0.0],
            [[0.73622596, 0.1292975], [0.5555515, 0.967837], [0.3889836, 0.31490302], [0.08661449, 0.91337407]]
        ]
    ]
    for input, expected in test_data:
        soft_input = jnp.array(input)
        hard_input = harden.harden_array(soft_input)
        soft_expected = jnp.array(expected)
        hard_expected = harden.harden_array(soft_expected)
        soft_result = soft_net.apply(soft_weights, soft_input)
        assert jnp.allclose(soft_result, soft_expected)
        hard_result = hard_net.apply(hard_weights, hard_input)
        assert jnp.array_equal(hard_result, hard_expected)
        symbolic_result = symbolic_net.apply(harden.harden_dict(soft_weights), harden.harden_array(soft_input))
        assert jnp.array_equal(eval(symbolic_result), hard_expected)

def test_train_not():
    rng = random.PRNGKey(0)
    soft_net, hard_net, symbolic_net = hard_not.NotLayer(layer_size=4)
    soft_weights = soft_net.init(rng, [1.0, 1.0])
    x = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    y = [
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ]
    input = jnp.array(x)
    output = jnp.array(y)
    # Train the not layer
    tx = optax.sgd(0.1)
    state = train_state.TrainState.create(apply_fn=jax.vmap(soft_net.apply, params=soft_weights, tx=tx)
    grad_fn = jax.jit(jax.value_and_grad(lambda params, x, y: jnp.mean((state.apply_fn(params, x) - y) ** 2)))
    for epoch in range(1, 50):
        loss, grads = grad_fn(state.params, input, output)
        state = state.apply_gradients(grads=grads)
    """
    # Test the not layer
    soft_weights = state.params
    hard_weights = harden.harden(soft_weights)
    hard_input = harden.harden_array(input)
    hard_expected = harden.harden_array(output)
    hard_result = hard_net.apply(hard_weights, hard_input)
    print("Hard result = ", hard_result)
    print("Hard expected = ", hard_expected)
    assert jnp.array_equal(hard_result, hard_expected)
    symbolic_result = symbolic_net.apply(harden.harden_dict(state.params), hard_input)
    print("Symbolic result = ", symbolic_result)
    print("Symbolic expected = ", hard_expected)
    assert jnp.array_equal(eval(symbolic_result), hard_expected)
    """

