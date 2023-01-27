from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import ml_collections
import numpy
import optax
from flax import linen as nn
from flax.training import train_state
from jax import lax, random
from tqdm import tqdm

from neurallogic import (
    hard_and,
    hard_majority,
    hard_not,
    hard_or,
    hard_xor,
    harden,
    harden_layer,
    neural_logic_net,
)
from tests import utils


def check_symbolic(nets, data, trained_state):
    x_training, y_training, x_test, y_test = get_train_and_test_data(data)
    _, hard, symbolic = nets
    _, test_loss, test_accuracy = apply_model_with_grad(trained_state, x_test, y_test)
    print(
        "soft_net: final test_loss: %.4f, final test_accuracy: %.2f"
        % (test_loss, test_accuracy * 100)
    )
    hard_weights = harden.hard_weights(trained_state.params)
    hard_trained_state = train_state.TrainState.create(
        apply_fn=hard.apply, params=hard_weights, tx=optax.sgd(1.0, 1.0)
    )
    hard_input = harden.harden(x_test)
    hard_test_accuracy = apply_hard_model(hard_trained_state, hard_input, y_test)
    print("hard_net: final test_accuracy: %.2f" % (hard_test_accuracy * 100))
    assert numpy.isclose(test_accuracy, hard_test_accuracy, atol=0.0001)
    if True:
        symbolic_weights = hard_weights  # utils.make_symbolic(hard_weights)
        symbolic_trained_state = train_state.TrainState.create(
            apply_fn=symbolic.apply, params=symbolic_weights, tx=optax.sgd(1.0, 1.0)
        )
        symbolic_input = hard_input.tolist()
        symbolic_test_accuracy = apply_hard_model(
            symbolic_trained_state, symbolic_input, y_test
        )
        print(
            "symbolic_net: final test_accuracy: %.2f" % (symbolic_test_accuracy * 100)
        )
        assert numpy.isclose(test_accuracy, symbolic_test_accuracy, atol=0.0001)
    if False:
        # CPU and GPU give different results, so we can't easily regress on a static symbolic expression
        symbolic_input = [f"x{i}" for i in range(len(hard_input[0].tolist()))]
        symbolic_output = symbolic.apply({"params": symbolic_weights}, symbolic_input)
        print("symbolic_output", symbolic_output[0][:10000])


class BinaryDropout(nn.Module):
    """Create a dropout layer.

    Note: When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
    to include an RNG seed named `'dropout'`. For example::

      model.apply({'params': params}, inputs=inputs, train=True, rngs={'dropout': dropout_rng})`

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng_collection: the rng collection name to use when requesting an rng key.
    """

    rate: float
    broadcast_dims: Sequence[int] = ()
    deterministic: Optional[bool] = None
    rng_collection: str = "dropout"

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          Masking means setting the input bits to 0.5.
          deterministic: if false the inputs are masked,
          whereas if true, no mask is applied and the inputs are returned
          as is.

        Returns:
          The masked inputs
        """
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)

        keep_prob = 1.0 - self.rate
        rng = self.make_rng(self.rng_collection)
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        # masked_values = jnp.ones_like(inputs, dtype=float) / 2.0
        masked_values = jnp.zeros_like(inputs, dtype=float)
        # masked_values = jnp.ones_like(inputs, dtype=float)
        # print(f"mask {mask.shape} {mask.dtype} {mask}")
        # print(
        #    f"masked_values {masked_values.shape} {masked_values.dtype} {masked_values}"
        # )
        # print(f"inputs {inputs.shape} {inputs.dtype} {inputs}")
        return lax.select(mask, inputs, masked_values)


num_features = 12
num_classes = 2


def get_data():
    # Create a path to the data directory
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    # Load the training data
    training_data = numpy.loadtxt(data_dir / "NoisyXORTrainingData.txt").astype(
        dtype=numpy.float32
    )
    # Load the test data
    test_data = numpy.loadtxt(data_dir / "NoisyXORTestData.txt").astype(
        dtype=numpy.float32
    )
    return training_data, test_data


# 100% test accuracy
def nln_100(type, x):
    x = hard_and.and_layer(type)(20)(x)
    x = hard_not.not_layer(type)(4)(x)
    x = x.ravel()
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


# 100% test accuracy
def nln(type, x, training: bool):
    x = hard_xor.xor_layer(type)(40)(x)
    # x = hard_not.not_layer(type)(1)(x)
    x = BinaryDropout(rate=0.5, deterministic=not training)(x)
    # x = x.ravel()
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


def batch_nln(type, x, training: bool):
    return jax.vmap(lambda x: nln(type, x, training))(x)


class TrainState(train_state.TrainState):
    dropout_rng: jax.random.KeyArray


def create_train_state(net, rng, dropout_rng, config):
    mock_input = jax.numpy.ones([1, num_features])
    soft_weights = net.init(rng, mock_input, training=False)["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    # tx = optax.yogi(config.learning_rate)
    # tx = optax.noisy_sgd(config.learning_rate, config.momentum)
    # tx = optax.adagrad(config.learning_rate) # investigate this one
    return TrainState.create(
        apply_fn=net.apply, params=soft_weights, tx=tx, dropout_rng=dropout_rng
    )


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def apply_model_with_grad_impl(state, features, labels, dropout_rng, training: bool):
    dropout_train_rng = jax.random.fold_in(key=dropout_rng, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            features,
            training=training,
            rngs={"dropout": dropout_train_rng},
        )
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = jax.numpy.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jax.numpy.mean(jax.numpy.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_with_grad_and_training(state, features, labels, dropout_rng):
    return apply_model_with_grad_impl(
        state, features, labels, dropout_rng, training=True
    )


@jax.jit
def apply_model_with_grad(state, features, labels, dropout_rng):
    return apply_model_with_grad_impl(
        state, features, labels, dropout_rng, training=False
    )


def train_epoch(state, features, labels, batch_size, rng, dropout_rng):
    train_ds_size = len(features)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(features))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_features = features[perm, ...]
        batch_labels = labels[perm, ...]
        grads, loss, accuracy = apply_model_with_grad_and_training(
            state, batch_features, batch_labels, dropout_rng
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = numpy.mean(epoch_loss)
    train_accuracy = numpy.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_train_and_test_data(data):
    training_data, test_data = data
    x_training = training_data[:, 0:num_features]  # Input features
    y_training = training_data[:, num_features]  # Target value
    x_test = test_data[:, 0:num_features]  # Input features
    y_test = test_data[:, num_features]  # Target value
    return x_training, y_training, x_test, y_test


def train_and_evaluate(
    init_rng, dropout_rng, net, data, config: ml_collections.ConfigDict
):
    state = create_train_state(net, init_rng, dropout_rng, config)
    x_training, y_training, x_test, y_test = data
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    for epoch in range(1, config.num_epochs + 1):
        init_rng, input_rng = jax.random.split(init_rng)
        state, train_loss, train_accuracy = train_epoch(
            state, x_training, y_training, config.batch_size, input_rng, dropout_rng
        )
        _, test_loss, test_accuracy = apply_model_with_grad(
            state, x_test, y_test, dropout_rng
        )
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            print(f"best_train_accuracy: {best_train_accuracy * 100:.2f}")
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy = test_accuracy
                print(f"best_test_accuracy: {best_test_accuracy * 100:.2f}")
            else:
                print(f"test_accuracy: {test_accuracy * 100:.2f}")
            print("\n")

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

    return state


def apply_hard_model(state, features, label):
    def logits_fn(params):
        return state.apply_fn({"params": params}, features, training=False)

    logits = logits_fn(state.params)
    if isinstance(logits, list):
        logits = jax.numpy.array(logits)
    logits *= 1.0
    accuracy = jax.numpy.mean(jax.numpy.argmax(logits, -1) == label)
    return accuracy


def apply_hard_model_to_data(state, features, labels):
    accuracy = 0
    for (image, label) in tqdm(zip(features, labels), total=len(features)):
        accuracy += apply_hard_model(state, image, label)
    return accuracy / len(features)


def get_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 0.01
    config.momentum = 0.9
    config.batch_size = 256
    config.num_epochs = 1000
    return config


def test_noisy_xor():
    rng = jax.random.PRNGKey(0)
    rng, int_rng, dropout_rng = jax.random.split(rng, 3)
    # Train net
    soft, hard, symbolic = neural_logic_net.net(
        lambda type, x, training: batch_nln(type, x, training)
    )
    x_training, y_training, x_test, y_test = get_train_and_test_data(get_data())
    print(soft.tabulate(rng, x_training[0:1], training=False))

    print(f"training_data.shape: {x_training.shape}")
    print(f"test_data.shape: {x_test.shape}")
    trained_state = train_and_evaluate(
        int_rng,
        dropout_rng,
        soft,
        (x_training, y_training, x_test, y_test),
        get_config(),
    )

    # Check symbolic net
    # _, hard, symbolic = neural_logic_net.net(lambda type, x: nln(type, x))
    # check_symbolic((soft, hard, symbolic), (training_data, test_data), trained_state)
