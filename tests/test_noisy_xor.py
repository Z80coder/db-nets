import sys
from pathlib import Path

import jax
import ml_collections
import numpy
import optax
import scipy
from flax import linen as nn
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
    harden,
    harden_layer,
    neural_logic_net,
    initialization,
)
from tests import utils

config.update("jax_enable_x64", True)


def check_symbolic(nets, data, trained_state, dropout_rng):
    x_training, y_training, x_test, y_test = data
    _, hard, symbolic = nets
    _, test_loss, test_accuracy = apply_model_with_grad(
        trained_state, x_test, y_test, dropout_rng
    )
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
    hard_test_accuracy = apply_hard_model_to_data(
        hard_trained_state, hard_input, y_test
    )
    print("hard_net: final test_accuracy: %.2f" % (hard_test_accuracy * 100))
    assert numpy.isclose(test_accuracy, hard_test_accuracy, atol=0.0001)
    if False:
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


"""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.01,
        peak_value=0.1,
        warmup_steps=100,
        decay_steps=config.num_epochs - 100,
        end_value=0.05,
    )
    tx = optax.chain(
        # optax.clip(1.0),
        #optax.adamw(learning_rate=schedule),
        optax.adam(learning_rate=schedule),
    )
"""
# with (4,1) error:
# mean: 98.04, sem: 0.42, min: 70.32, max: 100.00, 5%: 93.90, 95%: 100.00


# mean: 82.63, sem: 1.41, min: 48.86, max: 100.00, 5%: 59.70, 95%: 100.00
def nln_1(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)
    dtype = jax.numpy.float32
    layer_size = 48
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(8, dtype=dtype)(x)  # 48 x 8 = 384
    x = x.reshape((32, 12))  # 32 x 12 = 384
    x = hard_majority.majority_layer(type)()(x)
    x = x.reshape((4, 8))  # 4 x 8 = 32
    x = hard_majority.majority_layer(type)()(x)
    # Final majority should have reasonable width
    x = x.reshape((1, 4))  # 4 x 1 = 4
    x = hard_majority.majority_layer(type)()(x)
    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


"""
SGD
config.learning_rate = 2.0
config.momentum = 0.9
config.batch_size = 5000
config.num_epochs = 2000
"""


# mean: 91.06, sem: 1.15, min: 48.86, max: 99.16, 5%: 48.86, 95%: 98.28
def nln_2(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)
    dtype = jax.numpy.float32
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(64, dtype=dtype)(x)  # 32 x 64 = 2048
    x = x.reshape((64, 32))  # 64 x 32 = 2048
    x = hard_majority.majority_layer(type)()(x)
    x = x.reshape((1, 64))  # 1 x 64 = 64
    x = hard_majority.majority_layer(type)()(x)
    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


"""
SGD
config.learning_rate = 2.0
config.momentum = 0.9
config.batch_size = 5000
config.num_epochs = 2000
"""


# mean: 91.54, sem: 0.64, min: 48.86, max: 99.58, 5%: 83.09, 95%: 97.10
def nln_3(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float32
    layer_size = 64
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((1, layer_size * 16))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


"""
SGD
config.learning_rate = 1.0
config.momentum = 0.9
config.batch_size = 5000
config.num_epochs = 2500
"""


# mean: 91.84, sem: 0.98, min: 48.86, max: 99.66, 5%: 70.30, 95%: 98.79
def nln_4(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float32
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((1, layer_size * 16))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


# mean: 89.77, sem: 1.18, min: 48.80, max: 100.00, 5%: 67.40, 95%: 99.04
def nln_5(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float32
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(8, dtype=dtype)(x)

    x = x.reshape((1, layer_size * 8))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


# mean: 89.19, sem: 1.42, min: 48.86, max: 100.00, 5%: 51.14, 95%: 98.86
def nln_6(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float32
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((16, layer_size))
    x = hard_majority.majority_layer(type)()(x)

    x = hard_not.not_layer(type)(8, dtype=dtype)(x)
    x = x.reshape((1, 16 * 8))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


# mean: 89.34, sem: 0.74, min: 51.16, max: 99.28, 5%: 76.88, 95%: 96.98
def nln_7(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float64
    layer_size = 40
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(20, dtype=dtype)(x)

    x = x.reshape((1, layer_size * 20))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x



# mean: 94.15, sem: 0.44, min: 74.62, max: 100.00, 5%: 85.08, 95%: 99.10
# soft_majority variant
# optax.radam(learning_rate=0.005)
# config.batch_size = 5000
# config.num_epochs = 3000
# N.B. With normal soft majority we get
# mean: 83.82, sem: 1.75, min: 48.86, max: 100.00, 5%: 48.86, 95%: 99.42
# Hence new variant seems very effective in avoiding local minima
def nln_8(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float64
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((1, 16 * layer_size))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x



# mean: 94.51, sem: 0.37, min: 70.82, max: 100.00, 5%: 89.24, 95%: 98.15
def nln_9(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float64
    layer_size = 32
    x = hard_and.and_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_or.or_layer(type)(layer_size, dtype=dtype)(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((1, 16 * layer_size))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


"""
| Technique/Accuracy  | Mean           | 5 %ile  | 95 %ile | Min    | Max    |
| ------------------- | -------------- | ------- | ------- | ------ | ------ |
| Tsetlin             | 99.3 +/- 0.3   | 95.9    | 100.0   | 91.6   | 100.0  |
| dB                  | 96.5 +/- 0.2   | 93.2    | 99.5    | 87.9   | 100.0  |
| Neural network      | 95.4 +/- 0.5   | 90.1    | 98.6    | 88.2   | 99.9   |
| SVM                 | 58.0 +/- 0.3   | 56.4    | 59.2    | 55.4   | 66.5   |
| Naive Bayes         | 49.8 +/- 0.2   | 48.3    | 51.0    | 41.3   | 52.7   |
| Logistic regression | 49.8 +/- 0.3   | 47.8    | 51.1    | 41.1   | 53.1   |

Source: https://arxiv.org/pdf/1804.01508.pdf
"""

# mean: 96.52, sem: 0.23, min: 87.86, max: 100.00, 5%: 93.16, 95%: 99.51
def nln(type, x, training: bool):
    y = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, y], axis=0)

    dtype = jax.numpy.float64
    layer_size = 32
    x = hard_and.and_layer(type)(
        layer_size,
        dtype=dtype,
        weights_init=initialization.initialize_bernoulli(0.01, 0.1, 0.51),
    )(x)
    x = hard_or.or_layer(type)(
        layer_size,
        dtype=dtype,
        weights_init=initialization.initialize_bernoulli(0.99, 0.49, 0.9),
    )(x)
    x = hard_not.not_layer(type)(16, dtype=dtype)(x)

    x = x.reshape((1, 16 * layer_size))
    x = hard_majority.majority_layer(type)()(x)

    z = jax.vmap(lambda x: 1 - x)(x)
    x = jax.numpy.concatenate([x, z], axis=0)
    ########################################################
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

    """
    schedule = optax.linear_onecycle_schedule(
        transition_steps=3000,
        peak_value=1.0,
        pct_start=0.3,
        pct_final=0.85,
        div_factor=25.0,
        final_div_factor=1e4,
    )
    schedule = optax.linear_schedule(
        init_value=0.01,
        end_value=0.01,
        transition_steps=config.num_epochs,
        transition_begin=0,
    )
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.1,
        peak_value=2,
        warmup_steps=0,
        decay_steps=500,
        end_value=0.01,
    )
    tx = optax.chain(
        # optax.clip(1.0),
        # optax.adamw(learning_rate=0.01, weight_decay=0.0001),
        # optax.adam(learning_rate=config.learning_rate),
        # optax.adabelief(learning_rate=0.01),
        # optax.adamax(learning_rate=0.01),
        # optax.amsgrad(learning_rate=0.002),
        # optax.sm3(learning_rate=0.1),
        # optax.novograd(learning_rate=0.001),
        # optax.optimistic_gradient_descent(learning_rate=1),
        optax.radam(learning_rate=0.01),  # 0.02
        # optax.sgd(
        #    learning_rate=config.learning_rate, momentum=config.momentum, nesterov=False
        # ),
    )

    return TrainState.create(
        apply_fn=net.apply, params=soft_weights, tx=tx, dropout_rng=dropout_rng
    )


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def my_hinge_loss(predictor_outputs, targets):
    loss = jax.numpy.abs(predictor_outputs - targets)

    def hinge(x):
        return jax.numpy.where(x < 0.5, x, x)

    return jax.vmap(hinge)(loss)


def apply_model_with_grad_impl(state, features, labels, dropout_rng, training: bool):
    dropout_train_rng = jax.random.fold_in(key=dropout_rng, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            features,
            training=training,
            rngs={"dropout": dropout_train_rng},
        )
        one_hot = jax.nn.one_hot(labels, num_classes, dtype=jax.numpy.int32)
        loss = jax.numpy.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
            # optax.l2_loss(predictions=logits, targets=one_hot)
            # my_hinge_loss(logits, one_hot)
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
            # print(f"best_train_accuracy: {best_train_accuracy * 100:.2f}")
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy = test_accuracy
                # print(f"best_test_accuracy: {best_test_accuracy * 100:.2f}")
            # else:
            # print(f"test_accuracy: {test_accuracy * 100:.2f}")
            # print("\n")

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

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
    config.learning_rate = 0.001
    config.momentum = 0.9
    config.batch_size = 5000
    config.num_epochs = 1000
    return config


def test_noisy_xor():
    # Train net
    soft, _, _ = neural_logic_net.net(
        lambda type, x, training: batch_nln(type, x, training)
    )

    x_training, y_training, x_test, y_test = get_train_and_test_data(get_data())

    rng = jax.random.PRNGKey(0)
    print(soft.tabulate(rng, x_training[0:1], training=False))

    num_experiments = 1  # 100 for paper
    final_test_accuracies = []
    for i in range(num_experiments):
        rng, int_rng, dropout_rng = jax.random.split(rng, 3)
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
        # numpy.set_printoptions(threshold=sys.maxsize)
        # print(f"trained soft weights: {repr(trained_state.params)}")
        # hard_weights = harden.hard_weights(trained_state.params)
        # print(f"trained hard weights: {repr(hard_weights)}")

        # if final_test_accuracy < 0.85:
        #    print("Aborting due to poor performance")
        #    break

        # Check symbolic net
        """
        _, hard, symbolic = neural_logic_net.net(
            lambda type, x, training: nln(type, x, training)
        )
        check_symbolic(
            (soft, hard, symbolic),
            (x_training, y_training, x_test, y_test),
            trained_state,
            dropout_rng,
        )
        """
