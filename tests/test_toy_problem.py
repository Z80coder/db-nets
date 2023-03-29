from pathlib import Path

import jax
import ml_collections
import numpy
import optax
import scipy
from flax.training import train_state
from jax.config import config
from tqdm import tqdm

from neurallogic import (
    hard_and,
    hard_dropout,
    hard_majority,
    hard_masks,
    hard_not,
    hard_or,
    hard_xor,
    hard_count,
    harden,
    harden_layer,
    neural_logic_net,
    real_encoder,
    initialization,
    hard_vmap,
    hard_concatenate,
    symbolic_primitives
)
from tests import utils

config.update("jax_enable_x64", True)

"""
Temperature: 4 booleans, 1-hot vector
    0 high = very cold
    1 high = cold
    2 high = warm
    3 high = very warm
Outside?: 1 boolean
    0 = no
    1 = yes
Labels: 
    0 = wear t-shirt 
    1 = wear coat
"""

toy_data = 2
num_classes = 2
if toy_data == 1:
    num_features = 1
else:
    num_features = 5


def check_symbolic(nets, data, trained_state, dropout_rng):
    x_training, y_training, x_test, y_test = data
    _, hard, symbolic = nets
    _, test_loss, test_accuracy = apply_model_with_grad(trained_state, x_test, y_test, dropout_rng)
    print(
        "soft_net: final test_loss: %.4f, final test_accuracy: %.2f"
        % (test_loss, test_accuracy * 100)
    )
    hard_weights = harden.hard_weights(trained_state.params)
    hard_trained_state = TrainState.create(
        apply_fn=hard.apply,
        params=hard_weights,
        tx=optax.sgd(1.0, 1.0),
        dropout_rng=dropout_rng,
    )
    hard_input = harden.harden(x_test)
    hard_test_accuracy = apply_hard_model_to_data(hard_trained_state, hard_input, y_test)
    print("hard_net: final test_accuracy: %.2f" % (hard_test_accuracy * 100))
    assert numpy.isclose(test_accuracy, hard_test_accuracy, atol=0.0001)

    # CPU and GPU give different results, so we can't easily regress on a static symbolic expression
    if toy_data == 1:
        symbolic_input = ["outside"]
    else:
        symbolic_input = ["very-cold", "cold", "warm", "very-warm", "outside"]
    # This simply checks that the symbolic output can be generated
    symbolic_output = symbolic.apply({"params": hard_weights}, symbolic_input, training=False)
    print("symbolic_output: class 1", symbolic_output[0][:10000])
    print("symbolic_output: class 2", symbolic_output[1][:10000])


def nln_1(type, x, training: bool):
    dtype = jax.numpy.float32
    layer_size = 2
    x = hard_not.not_layer(type)(layer_size)(x)
    x = x.ravel()
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = hard_majority.majority_layer(type)()(x)
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

"""
Class 1:
lax_reference.ge(lax_reference.sum((0, numpy.logical_not(numpy.logical_xor(lax_reference.ne(x0, 0), False)))), 1)

is equivalent to:

sum(
    (
        0, 
        ! xor(x0 != 0, False)
    )
) >= 1

is equivalent to:

! xor(x0 != 0, False) >= 1

is equivalent to:

! x0

Class 2:
class 2 lax_reference.ge(lax_reference.sum((0, numpy.logical_not(numpy.logical_xor(lax_reference.ne(x0, 0), True)))), 1)

is equivalent to:

! xor(x0 != 0, True) >= 1

is equivalent to:

x0

Therefore learned class prediction is [!x, x]
"""

def nln_2(type, x, training: bool):
    dtype = jax.numpy.float32
    x = hard_not.not_layer(type)(8)(x)
    x = x.ravel()
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = hard_majority.majority_layer(type)()(x)
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

def nln(type, x, training: bool):
    if toy_data == 1:
        return nln_1(type, x, training)
    else:
        return nln_2(type, x, training)

def batch_nln(type, x, training: bool):
    return jax.vmap(lambda x: nln(type, x, training))(x)


class TrainState(train_state.TrainState):
    dropout_rng: jax.random.KeyArray


def create_train_state(net, rng, dropout_rng, config):
    mock_input = jax.numpy.ones([1, num_features])
    soft_weights = net.init(rng, mock_input, training=False)["params"]
    tx = optax.radam(learning_rate=config.learning_rate)
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
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy = test_accuracy
        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )
        if train_accuracy == 1.0 and test_accuracy == 1.0:
            break

    # return trained state and final test_accuracy
    return state, test_accuracy


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
    for image, label in tqdm(zip(features, labels), total=len(features)):
        accuracy += apply_hard_model(state, image, label)
    return accuracy / len(features)


def get_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 0.01
    config.momentum = 0.9
    if toy_data == 1:
        config.batch_size = 16
    else:
        config.batch_size = 48
    config.num_epochs = 1000
    return config

def get_toy_data():
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    if toy_data == 1:
        data = numpy.loadtxt(data_dir / "toy_data_1.txt").astype(dtype=numpy.int32)
    else:
        data = numpy.loadtxt(data_dir / "toy_data_2.txt").astype(dtype=numpy.int32)
    features = data[:, 0:num_features]  # Input features
    labels = data[:, num_features]  # Target value
    return features, labels


def train_test_split(features, labels, rng, test_size=0.2):
    rng, split_rng = jax.random.split(rng)
    train_size = int(len(features) * (1 - test_size))
    train_idx = jax.random.permutation(split_rng, len(features))[:train_size]
    test_idx = jax.random.permutation(split_rng, len(features))[train_size:]
    return (
        features[train_idx],
        features[test_idx],
        labels[train_idx],
        labels[test_idx],
    )

@pytest.mark.skip(reason="temporarily off")
def test_toy():
    # Train net
    features, labels = get_toy_data()
    soft, hard, symbolic = neural_logic_net.net(
        lambda type, x, training: batch_nln(type, x, training)
    )

    rng = jax.random.PRNGKey(0)
    print(soft.tabulate(rng, features[0:1], training=False))

    num_experiments = 1
    final_test_accuracies = []
    for i in range(num_experiments):
        # Split features and labels into 80% training and 20% test
        rng, int_rng, dropout_rng = jax.random.split(rng, 3)
        x_training, x_test, y_training, y_test = train_test_split(
            features, labels, rng, test_size=0.2
        )
        trained_state, final_test_accuracy = train_and_evaluate(
            int_rng,
            dropout_rng,
            soft,
            (x_training, y_training, x_test, y_test),
            get_config(),
        )
        final_test_accuracies.append(final_test_accuracy)
        print(f"{i}: final test accuracy: {final_test_accuracy * 100:.2f}")
        # print mean, standard error of the mean, min, max, lowest 5%, highest 5% of final test accuracies
        print(
            f"mean: {numpy.mean(final_test_accuracies) * 100:.2f}, "
            f"sem: {scipy.stats.sem(final_test_accuracies) * 100:.2f}, "
            f"min: {numpy.min(final_test_accuracies) * 100:.2f}, "
            f"max: {numpy.max(final_test_accuracies) * 100:.2f}, "
            f"5%: {numpy.percentile(final_test_accuracies, 5) * 100:.2f}, "
            f"95%: {numpy.percentile(final_test_accuracies, 95) * 100:.2f}"
        )

    # Check symbolic net
    _, hard, symbolic = neural_logic_net.net(lambda type, x, training: nln(type, x, training))
    check_symbolic((soft, hard, symbolic), (x_training, y_training, x_test, y_test), trained_state, dropout_rng)
