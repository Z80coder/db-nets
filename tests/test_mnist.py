import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import ml_collections
from neurallogic import (hard_and, hard_not, hard_or, harden, harden_layer,
                         neural_logic_net, primitives)
import optax


"""
MNIST test.

Executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

def nln(type, x):
  x = primitives.nl_ravel(type)(x)
  x = hard_or.or_layer(type)(100, nn.initializers.uniform(1.0), dtype=jnp.float32)(x) # >=1500 need for >98% accuracy
  x = hard_not.not_layer(type)(10, dtype=jnp.float32)(x)
  x = primitives.nl_ravel(type)(x) 
  x = harden_layer.harden_layer(type)(x)
  x = primitives.nl_reshape(type)((10, 100))(x)
  x = primitives.nl_sum(type)(-1)(x)
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
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
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
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy

def get_datasets():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  # convert the floating point values in [0,1] to binary values in {0,1}
  train_ds['image'] = jnp.round(train_ds['image'])
  test_ds['image'] = jnp.round(test_ds['image'])
  return train_ds, test_ds

def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  # soft = CNN()
  soft, hard, symbolic = neural_logic_net.net(batch_nln)
  mock_input = jnp.ones([1, 28, 28, 1])
  soft_weights = soft.init(rng, mock_input)['params']
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(apply_fn=soft.apply, params=soft_weights, tx=tx)

def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    config.batch_size,
                                                    input_rng)
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'])

    print(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))
           
    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  return state

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # config for CNN
  config.learning_rate = 0.01
  # config for NLN
  config.learning_rate = 0.1
  
  # Always commit with num_epochs = 1 for short test time
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 1
  return config

def test_mnist():
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], "GPU")

  # Define training configuration.
  config = get_config()

  rng = jax.random.PRNGKey(0)
  state = create_train_state(rng, config)

  # Create a temporary directory where tensorboard metrics are written.
  workdir = "./mnist_metrics"
  train_and_evaluate(config=config, workdir=workdir)

