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

def check_symbolic(nets, data, trained_state):
    x_training, y_training, x_test, y_test = data
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


binary_iris = True
num_features = 16 if binary_iris else 4
num_classes = 3


def get_iris_data():
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    data = numpy.loadtxt(
        data_dir / "iris.data",
        delimiter=",",
        dtype={
            "names": (
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "class",
            ),
            "formats": ("f4", "f4", "f4", "f4", "U15"),
        },
    )
    features = numpy.array([list(d)[:4] for d in data])
    # Normalise each feature column to be in the range [0, 1]
    features = (features - features.min(axis=0)) / (
        features.max(axis=0) - features.min(axis=0)
    )
    labels = numpy.array(
        [
            0
            if d[num_features] == "Iris-setosa"
            else 1
            if d[num_features] == "Iris-versicolor"
            else 2
            for d in data
        ]
    )
    return features, labels


def get_binary_iris_data():
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    data = numpy.loadtxt(data_dir / "BinaryIrisData.txt").astype(dtype=numpy.int32)
    features = data[:, 0:num_features]  # Input features
    labels = data[:, num_features]  # Target value
    return features, labels


# overfitting model: 100% training accuracy
def nln_iris(type, x, training: bool):
    input_size = x.shape[0]
    bits_per_feature = 10
    x = real_encoder.real_encoder_layer(type)(bits_per_feature)(x)
    x = x.ravel()
    dtype = jax.numpy.float32
    mask_layer_size = 120
    x = hard_masks.mask_to_true_margin_layer(type)(mask_layer_size, dtype=dtype)(x)
    x = x.reshape((mask_layer_size, input_size * bits_per_feature))
    x = hard_majority.majority_layer(type)()(x)
    x = hard_not.not_layer(type)(18)(x)
    x = x.ravel()
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


"""
| Technique/Accuracy | Mean           | 5 %ile  | 95 %ile | Min    | Max    |
| ------------------ | -------------- | ------- | ------- | ------ | ------ |
| Tsetlin            | 95.0 +/- 0.2   | 86.7    | 100.0   | 80.0   | 100.0  |
| dB                 | 94.2 +/- 0.1   | 86.7    | 100.0   | 80.0   | 100.0  |
| Neural network     | 93.8 +/- 0.2   | 86.7    | 100.0   | 80.0   | 100.0  |
| SVM                | 93.6 +/- 0.3   | 86.7    | 100.0   | 76.7   | 100.0  |
| Naive Bayes        | 91.6 +/- 0.3   | 83.3    | 96.7    | 70.0   | 100.0  |

Source: https://arxiv.org/pdf/1804.01508.pdf
"""

# TODO: implement count layer, k-high neuron, and multi-label classification
# to avoid the need for the harden layer

# Using majority without margin
# mean: 94.18, sem: 0.13, min: 80.00, max: 100.00, 5%: 86.67, 95%: 100.00
# Using majority with margin
# mean: 93.95, sem: 0.13, min: 76.67, max: 100.00, 5%: 86.67, 95%: 100.00
def nln_binary_iris_1(type, x, training: bool):
    dtype = jax.numpy.float32
    x = hard_masks.mask_to_true_layer(type)(120, dtype=dtype)(x)
    x = hard_majority.majority_layer(type)()(x)
    x = hard_dropout.hard_dropout(type)(
        rate=0.25,
        dropout_value=0.0,
        deterministic=not training,
        dtype=dtype,
    )(x)
    ########################################################
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

def nln_binary_iris(type, x, training: bool):
    dtype = jax.numpy.float64
    y = hard_vmap.vmap(type)((lambda x: 1 - x, lambda x: 1 - x, lambda x: symbolic_primitives.symbolic_not(x)))(x)
    x = hard_concatenate.concatenate(type)([x, y], 0)
    layer_size = 16
    x = hard_and.and_layer(type)(
        layer_size,
        dtype=dtype,
        weights_init=initialization.initialize_bernoulli(0.01, 0.3, 0.501),
    )(x)
    x = x.ravel()
    x = x.reshape((num_classes - 1, int(x.shape[0] / (num_classes - 1))))
    x = hard_majority.majority_layer(type)()(x)
    ########################################################
    x = jax.numpy.array([x]) # TODO: shouldn't need to do this
    x = hard_count.count_layer(type)()(x)
    x = x.ravel()
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

def batch_nln_iris(type, x, training: bool):
    return jax.vmap(lambda x: nln_iris(type, x, training))(x)


def batch_nln_binary_iris(type, x, training: bool):
    return jax.vmap(lambda x: nln_binary_iris(type, x, training))(x)


class TrainState(train_state.TrainState):
    dropout_rng: jax.random.KeyArray


def create_train_state(net, rng, dropout_rng, config):
    mock_input = jax.numpy.ones([1, num_features])
    soft_weights = net.init(rng, mock_input, training=False)["params"]
    #tx = optax.sgd(config.learning_rate, config.momentum)
    tx = optax.radam(learning_rate=config.learning_rate)
    return TrainState.create(
        apply_fn=net.apply, params=soft_weights, tx=tx, dropout_rng=dropout_rng
    )


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def my_loss(predictions, targets):
    return jax.vmap(lambda x: jax.numpy.where(x < 0.45, 0.0, x*x))(predictions - targets)

def my_loss(predictions, targets):
    x = predictions - targets
    return x*x

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
            #optax.l2_loss(logits, one_hot)
            #my_loss(logits, one_hot)
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
            # print(f"epoch: {epoch}")
            # print(f"\tbest_train_accuracy: {best_train_accuracy * 100:.2f}")
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy = test_accuracy
                # print(f"\tbest_test_accuracy: {best_test_accuracy * 100:.2f}")
            # else:
            #    print(f"\ttest_accuracy: {test_accuracy * 100:.2f}")
            # print("\n")

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

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
    config.learning_rate = 0.01 # sgd = 0.1
    config.momentum = 0.9
    config.batch_size = 120
    config.num_epochs = 4000 # 20000  # 500 for paper
    return config


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
def test_iris():
    # Train net
    if binary_iris:
        features, labels = get_binary_iris_data()
        soft, hard, symbolic = neural_logic_net.net(
            lambda type, x, training: batch_nln_binary_iris(type, x, training)
        )
    else:
        features, labels = get_iris_data()
        soft, hard, symbolic = neural_logic_net.net(
            lambda type, x, training: batch_nln_iris(type, x, training)
        )

    rng = jax.random.PRNGKey(0)
    print(soft.tabulate(rng, features[0:1], training=False))

    num_experiments = 1000  # 1000 for paper
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
    # _, hard, symbolic = neural_logic_net.net(lambda type, x: nln(type, x))
    # check_symbolic((soft, hard, symbolic), (x_training, y_training, x_test, y_test), trained_state)
