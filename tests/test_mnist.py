import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from jax.config import config
from matplotlib import pyplot as plt
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
    real_encoder,
)

# Uncomment to debug NaNs
# config.update("jax_debug_nans", True)

"""
MNIST test.

Executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# TODO: experiment in ipython notebook with different values for these
"""
def nln(type, x, width):
    #x = x.reshape((-1, 1))
    #re = real_encoder.real_encoder_layer(type)(3)
    #x = jax.vmap(re, 0)(x)
    #x = x.ravel()
    #x = hard_or.or_layer(type)(width, nn.initializers.uniform(1.0), dtype=jax.numpy.float16)(x)
    x = hard_or.or_layer(type)(width, nn.initializers.uniform(1.0), dtype=jax.numpy.float16)(x)
    x = hard_not.not_layer(type)(10)(x)
    x = x.ravel()  # flatten the outputs of the not layer
    # harden the outputs of the not layer
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((10, width))  # reshape to 10 ports, 100 bits each
    x = x.sum(-1)  # sum the 100 bits in each port
    return x
"""


def nln(type, x):
    num_classes = 10

    x = hard_or.or_layer(type)(
        1800, nn.initializers.uniform(1.0), dtype=jax.numpy.float16
    )(x)
    x = hard_not.not_layer(type)(1, dtype=jax.numpy.float16)(x)
    x = x.ravel()
    x = harden_layer.harden_layer(type)(x)
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x


def batch_nln(type, x):
    return jax.vmap(lambda x: nln(type, x))(x)


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model_with_grad(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model_with_grad(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    # XXXX
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    # TODO: we don't need to do this even when we don't use the real encoder
    # Use grayscale information
    # Convert the floating point values in [0,1] to binary values in {0,1}
    # train_ds["image"] = jnp.round(train_ds["image"])
    # test_ds["image"] = jnp.round(test_ds["image"])
    return train_ds, test_ds


def show_img(img, ax=None, title=None):
    """Shows a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img.reshape(28, 28), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def show_img_grid(imgs, titles):
    """Shows a grid of images."""
    n = int(np.ceil(len(imgs) ** 0.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axs[i // n][i % n], title)


def create_train_state(net, rng, config):
    """Creates initial `TrainState`."""
    # for CNN
    # mock_input = jnp.ones([1, 28, 28, 1])
    # for NLN
    mock_input = jnp.ones([1, 28 * 28])
    soft_weights = net.init(rng, mock_input)["params"]
    # tx = optax.sgd(config.learning_rate, config.momentum)
    # tx = optax.noisy_sgd(config.learning_rate, config.momentum)
    tx = optax.yogi(config.learning_rate)
    return train_state.TrainState.create(apply_fn=net.apply, params=soft_weights, tx=tx)


def train_and_evaluate(
    net, datasets, config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
    train_dataset, test_dataset = datasets
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(net, init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_dataset, config.batch_size, input_rng
        )
        _, test_loss, test_accuracy = apply_model_with_grad(
            state, test_dataset["image"], test_dataset["label"]
        )

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    return state


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # config for CNN
    config.learning_rate = 0.01
    # config for NLN
    # config.learning_rate = 0.1
    config.learning_rate = 0.01

    # Always commit with num_epochs = 1 for short test time
    config.momentum = 0.9
    config.batch_size = 128
    # config.num_epochs = 2
    config.num_epochs = 1000
    return config


def apply_hard_model(state, image, label):
    def logits_fn(params):
        return state.apply_fn({"params": params}, image)

    logits = logits_fn(state.params)
    if isinstance(logits, list):
        logits = jnp.array(logits)
    logits *= 1.0
    accuracy = jnp.mean(jnp.argmax(logits, -1) == label)
    return accuracy


def apply_hard_model_to_images(state, images, labels):
    accuracy = 0
    for (image, label) in tqdm(zip(images, labels), total=len(images)):
        accuracy += apply_hard_model(state, image, label)
    return accuracy / len(images)


def check_symbolic(nets, datasets, trained_state):
    _, test_ds = datasets
    _, hard, symbolic = nets
    _, test_loss, test_accuracy = apply_model_with_grad(
        trained_state, test_ds["image"], test_ds["label"]
    )
    print(
        "soft_net: final test_loss: %.4f, final test_accuracy: %.2f"
        % (test_loss, test_accuracy * 100)
    )
    hard_weights = harden.hard_weights(trained_state.params)
    hard_trained_state = train_state.TrainState.create(
        apply_fn=hard.apply, params=hard_weights, tx=optax.sgd(1.0, 1.0)
    )
    hard_input = harden.harden(test_ds["image"])
    hard_test_accuracy = apply_hard_model_to_images(
        hard_trained_state, hard_input, test_ds["label"]
    )
    print("hard_net: final test_accuracy: %.2f" % (hard_test_accuracy * 100))
    assert np.isclose(test_accuracy, hard_test_accuracy, atol=0.0001)
    # TODO: activate these checks
    if False:
        # It takes too long to compute this
        symbolic_weights = harden.symbolic_weights(trained_state.params)
        symbolic_trained_state = train_state.TrainState.create(
            apply_fn=symbolic.apply, params=symbolic_weights, tx=optax.sgd(1.0, 1.0)
        )
        symbolic_input = hard_input.tolist()
        symbolic_test_accuracy = apply_hard_model_to_images(
            symbolic_trained_state, symbolic_input, test_ds["label"]
        )
        print(
            "symbolic_net: final test_accuracy: %.2f" % (symbolic_test_accuracy * 100)
        )
        assert np.isclose(test_accuracy, symbolic_test_accuracy, atol=0.0001)
    if False:
        # CPU and GPU give different results, so we can't easily regress on a static symbolic expression
        symbolic_input = [f"x{i}" for i in range(len(hard_input[0].tolist()))]
        symbolic_output = symbolic.apply({"params": symbolic_weights}, symbolic_input)
        print("symbolic_output", symbolic_output[0][:10000])


@pytest.mark.skip(reason="temporarily off")
def test_mnist():
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")

    # Define training configuration.
    config = get_config()

    # Define the model.
    # soft = CNN()
    # width = 800
    soft, hard, _ = neural_logic_net.net(lambda type, x: batch_nln(type, x))

    # Get the MNIST dataset.
    train_ds, test_ds = get_datasets()
    # If we're using a NLN then flatten the images
    train_ds["image"] = jnp.reshape(train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(test_ds["image"], (test_ds["image"].shape[0], -1))

    print(soft.tabulate(jax.random.PRNGKey(0), train_ds["image"][0:1]))
    # TODO: fix the size of this
    # print(hard.tabulate(jax.random.PRNGKey(0), harden.harden(train_ds["image"][0:1])))

    # Train and evaluate the model.
    trained_state = train_and_evaluate(
        soft, (train_ds, test_ds), config=config, workdir="./mnist_metrics"
    )

    # Check symbolic net
    _, hard, symbolic = neural_logic_net.net(lambda type, x: nln(type, x))
    check_symbolic((soft, hard, symbolic), (train_ds, test_ds), trained_state)
