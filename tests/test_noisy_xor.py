from pathlib import Path
import ml_collections
import numpy
import optax
from flax.training import train_state
from flax import linen as nn
import jax

from neurallogic import (
    neural_logic_net,
    hard_not,
    hard_or,
    hard_and,
    hard_xor,
    hard_majority,
    harden_layer,
)

num_features = 12
num_classes = 2


def get_data():
    # Create a path to the data directory
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    # Load the training data
    training_data = numpy.loadtxt(data_dir / "NoisyXORTrainingData.txt").astype(
        dtype=numpy.int32
    )
    # Load the test data
    test_data = numpy.loadtxt(data_dir / "NoisyXORTestData.txt").astype(
        dtype=numpy.int32
    )
    return training_data, test_data


# 89% test accuracy
def nln_89(type, x):
    x = hard_and.and_layer(type)(20)(x)
    x = hard_not.not_layer(type)(5)(x)
    x = x.ravel()
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


# 100% test accuracy
def nln(type, x):
    x = hard_and.and_layer(type)(20)(x)
    x = hard_not.not_layer(type)(4)(x)
    x = x.ravel()
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


def batch_nln(type, x):
    return jax.vmap(lambda x: nln(type, x))(x)


def create_train_state(net, rng, config):
    mock_input = jax.numpy.ones([1, num_features])
    soft_weights = net.init(rng, mock_input)["params"]
    # tx = optax.sgd(config.learning_rate, config.momentum)
    tx = optax.yogi(config.learning_rate)
    return train_state.TrainState.create(apply_fn=net.apply, params=soft_weights, tx=tx)


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def apply_model_with_grad(state, features, labels):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, features)
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = jax.numpy.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jax.numpy.mean(jax.numpy.argmax(logits, -1) == labels)
    return grads, loss, accuracy


def train_epoch(state, features, labels, batch_size, rng):
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
        grads, loss, accuracy = apply_model_with_grad(
            state, batch_features, batch_labels
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = numpy.mean(epoch_loss)
    train_accuracy = numpy.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def train_and_evaluate(net, datasets, config: ml_collections.ConfigDict):
    training_data, test_data = datasets
    x_training = training_data[:, 0:num_features]  # Input features
    y_training = training_data[:, num_features]  # Target value
    x_test = test_data[:, 0:num_features]  # Input features
    y_test = test_data[:, num_features]  # Target value

    rng = jax.random.PRNGKey(0)
    print(net.tabulate(rng, x_training[0:1]))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(net, init_rng, config)

    best_test_accuracy = 0.0
    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, x_training, y_training, config.batch_size, input_rng
        )
        _, test_loss, test_accuracy = apply_model_with_grad(state, x_test, y_test)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )
        print(f"best_test_accuracy: {best_test_accuracy * 100:.2f}")

    return state


def get_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 0.01
    config.momentum = 0.9
    config.batch_size = 256
    config.num_epochs = 1000
    return config


def test_noisy_xor():
    soft, hard, _ = neural_logic_net.net(lambda type, x: batch_nln(type, x))
    training_data, test_data = get_data()
    trained_state = train_and_evaluate(soft, (training_data, test_data), get_config())
