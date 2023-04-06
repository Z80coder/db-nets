import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state
from jax.config import config
from tqdm import tqdm

from neurallogic import (
    hard_and,
    hard_majority,
    hard_not,
    hard_or,
    hard_xor,
    hard_masks,
    harden,
    harden_layer,
    neural_logic_net,
    real_encoder,
    hard_dropout,
    initialization,
    hard_count,
    hard_vmap,
    symbolic_primitives,
    hard_concatenate
)

# Uncomment to debug NaNs
# config.update("jax_debug_nans", True)


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


def check_symbolic(nets, datasets, trained_state, dropout_rng):
    _, test_ds = datasets
    _, hard, symbolic = nets
    _, test_loss, test_accuracy = apply_model_with_grad(
        trained_state, test_ds["image"], test_ds["label"], dropout_rng
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


# about 95% training, 93-4% test
# batch size 6000
def nln_1(type, x, training: bool):
    input_size = 784
    mask_layer_size = 60
    dtype = jax.numpy.float32
    x = hard_masks.mask_to_true_layer(type)(mask_layer_size, dtype=dtype,
        weights_init=initialization.initialize_bernoulli(0.01, 0.3, 0.501))(x)
    x = x.reshape((2940, 16)) 
    x = hard_majority.majority_layer(type)()(x)
    x = hard_not.not_layer(type)(20, weights_init=nn.initializers.uniform(1.0), dtype=dtype)(x)
    x = x.ravel()
    ##############################
    x = harden_layer.harden_layer(type)(x)
    num_classes = 10
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

def nln(type, x, training: bool):
    input_size = 784
    mask_layer_size = 200
    dtype = jax.numpy.float32
    x = hard_masks.mask_to_true_layer(type)(mask_layer_size, dtype=dtype,
        weights_init=initialization.initialize_bernoulli(0.01, 0.3, 0.501))(x)
    x = x.reshape((9800, 16)) 
    x = hard_majority.majority_layer(type)()(x)
    x = hard_not.not_layer(type)(20, weights_init=nn.initializers.uniform(1.0), dtype=dtype)(x)
    x = x.ravel()
    ##############################
    x = harden_layer.harden_layer(type)(x)
    num_classes = 10
    x = x.reshape((num_classes, int(x.shape[0] / num_classes)))
    x = x.sum(-1)
    return x

def batch_nln(type, x, training: bool):
    return jax.vmap(lambda x: nln(type, x, training))(x)


def apply_model_with_grad_impl(state, images, labels, dropout_rng, training: bool):
    dropout_train_rng = jax.random.fold_in(key=dropout_rng, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            images,
            training=training,
            rngs={"dropout": dropout_train_rng},
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_with_grad_and_training(state, images, labels, dropout_rng):
    return apply_model_with_grad_impl(state, images, labels, dropout_rng, training=True)


@jax.jit
def apply_model_with_grad(state, images, labels, dropout_rng):
    return apply_model_with_grad_impl(
        state, images, labels, dropout_rng, training=False
    )


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng, dropout_rng):
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
        grads, loss, accuracy = apply_model_with_grad_and_training(
            state, batch_images, batch_labels, dropout_rng
        )
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
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    # Convert the floating point values in [0,1] to binary values in {0,1}
    # If the float value is > 0.3 then we convert to 1, otherwise 0
    train_ds["image"] = jnp.where(train_ds["image"] > 0.3, 1.0, 0.0)
    test_ds["image"] = jnp.where(test_ds["image"] > 0.3, 1.0, 0.0)
    #train_ds["image"] = jnp.round(train_ds["image"])
    #test_ds["image"] = jnp.round(test_ds["image"])
    return train_ds, test_ds


def show_img(img, ax=None, title=None):
    """Shows a single image."""
    """
    if ax is None:
        ax = plt.gca()
    ax.imshow(img.reshape(28, 28), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    """

def show_img_grid(imgs, titles):
    """Shows a grid of images."""
    """
    n = int(np.ceil(len(imgs) ** 0.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axs[i // n][i % n], title)
    """

class TrainState(train_state.TrainState):
    dropout_rng: jax.random.KeyArray


def create_train_state(net, rng, dropout_rng, config):
    # for CNN: mock_input = jnp.ones([1, 28, 28, 1])
    mock_input = jnp.ones([1, 28 * 28])
    soft_weights = net.init(rng, mock_input, training=False)["params"]
    # tx = optax.yogi(config.learning_rate) # for nln_2
    tx = optax.radam(config.learning_rate)
    return TrainState.create(
        apply_fn=net.apply, params=soft_weights, tx=tx, dropout_rng=dropout_rng
    )


def train_and_evaluate(
    init_rng,
    dropout_rng,
    net,
    datasets,
    config: ml_collections.ConfigDict,
    workdir: str,
):
    state = create_train_state(net, init_rng, dropout_rng, config)
    train_dataset, test_dataset = datasets

    for epoch in range(1, config.num_epochs + 1):
        init_rng, input_rng = jax.random.split(init_rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_dataset, config.batch_size, input_rng, dropout_rng
        )
        _, test_loss, test_accuracy = apply_model_with_grad(
            state, test_dataset["image"], test_dataset["label"], dropout_rng
        )

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

    return state


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


def get_config():
    config = ml_collections.ConfigDict()
    # config for CNN: config.learning_rate = 0.01
    config.learning_rate = 0.01
    config.momentum = 0.9
    config.batch_size = 3000 # 6000 # 128
    config.num_epochs = 5000
    return config


#@pytest.mark.skip(reason="temporarily off")
def test_mnist():
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")

    rng = jax.random.PRNGKey(0)
    rng, int_rng, dropout_rng = jax.random.split(rng, 3)

    # soft = CNN()
    soft, _, _ = neural_logic_net.net(
        lambda type, x, training: batch_nln(type, x, training)
    )

    train_ds, test_ds = get_datasets()
    # If we're using a NLN then flatten the images
    train_ds["image"] = jnp.reshape(train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(test_ds["image"], (test_ds["image"].shape[0], -1))

    print(soft.tabulate(rng, train_ds["image"][0:1], training=False))

    # TODO: 50 experiments

    # Train and evaluate the model.
    trained_state = train_and_evaluate(
        int_rng,
        dropout_rng,
        soft,
        (train_ds, test_ds),
        config=get_config(),
        workdir="./mnist_metrics",
    )

    # Check symbolic net
    #_, hard, symbolic = neural_logic_net.net(lambda type, x: nln(type, x, False))
    #check_symbolic(
    #    (soft, hard, symbolic), (train_ds, test_ds), trained_state, dropout_rng
    #)

