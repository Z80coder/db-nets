from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import ml_collections
from neurallogic import (hard_not, hard_or, harden, harden_layer,
                         neural_logic_net, primitives)
import optax


"""
MNIST test.

Executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

def nln(type, x):
    x = hard_or.or_layer(type)(1700, nn.initializers.uniform(1.0), dtype=jnp.float32)(x) # >=1500 need for >98% accuracy
    x = hard_not.not_layer(type)(10, dtype=jnp.float32)(x)
    x = primitives.nl_ravel(type)(x) # flatten the outputs of the not layer
    x = harden_layer.harden_layer(type)(x) # harden the outputs of the not layer
    x = primitives.nl_reshape(type)((10, 1700))(x) # reshape to 10 ports, 100 bits each
    x = primitives.nl_sum(type)(-1)(x) # sum the 100 bits in each port
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
      grads, loss, accuracy = apply_model_with_grad(state, batch_images, batch_labels)
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
    # Convert the floating point values in [0,1] to binary values in {0,1}
    train_ds['image'] = jnp.round(train_ds['image'])
    test_ds['image'] = jnp.round(test_ds['image'])
    return train_ds, test_ds

def create_train_state(net, rng, config):
    """Creates initial `TrainState`."""
    # for CNN
    # mock_input = jnp.ones([1, 28, 28, 1])
    # for NLN
    mock_input = jnp.ones([1, 28 * 28])
    soft_weights = net.init(rng, mock_input)['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=net.apply, params=soft_weights, tx=tx)

def train_and_evaluate(net, datasets, config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = datasets
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(net, init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
      rng, input_rng = jax.random.split(rng)
      state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                      config.batch_size,
                                                      input_rng)
      _, test_loss, test_accuracy = apply_model_with_grad(state, test_ds['image'],
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
    config.num_epochs = 2
    return config

def apply_hard_model(state, image, label):
    def logits_fn(params):
      return state.apply_fn({'params': params}, image)

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
    _, test_loss, test_accuracy = apply_model_with_grad(trained_state, test_ds['image'], test_ds['label'])
    print('soft_net: final test_loss: %.4f, final test_accuracy: %.2f' % (test_loss, test_accuracy * 100))
    hard_weights = harden.hard_weights(trained_state.params)
    hard_trained_state = train_state.TrainState.create(apply_fn=hard.apply, params=hard_weights, tx=optax.sgd(1.0, 1.0))
    hard_input = harden.harden(test_ds['image'])
    hard_test_accuracy = apply_hard_model_to_images(hard_trained_state, hard_input, test_ds['label'])
    print('hard_net: final test_accuracy: %.2f' % (hard_test_accuracy * 100))
    assert np.isclose(test_accuracy, hard_test_accuracy, atol=0.0001)
    symbolic_weights = harden.symbolic_weights(trained_state.params)
    if False:
      # It takes too long to compute this
      symbolic_trained_state = train_state.TrainState.create(apply_fn=symbolic.apply, params=symbolic_weights, tx=optax.sgd(1.0, 1.0))
      symbolic_input = hard_input.tolist()
      symbolic_test_accuracy = apply_hard_model_to_images(symbolic_trained_state, symbolic_input, test_ds['label'])
      print('symbolic_net: final test_accuracy: %.2f' % (symbolic_test_accuracy * 100))
      assert(np.isclose(test_accuracy, symbolic_test_accuracy, atol=0.0001))
    symbolic_input = [f"x{i}" for i in range(len(hard_input[0].tolist()))]
    symbolic_output = symbolic.apply({'params': symbolic_weights}, symbolic_input)
    assert symbolic_output[0][:10000] == '((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((not(((x0 and False) or (x1 and True) or (x2 and True) or (x3 and True) or (x4 and True) or (x5 and True) or (x6 and False) or (x7 and True) or (x8 and True) or (x9 and True) or (x10 and True) or (x11 and True) or (x12 and True) or (x13 and False) or (x14 and False) or (x15 and True) or (x16 and False) or (x17 and True) or (x18 and True) or (x19 and True) or (x20 and False) or (x21 and True) or (x22 and True) or (x23 and False) or (x24 and False) or (x25 and False) or (x26 and False) or (x27 and False) or (x28 and True) or (x29 and True) or (x30 and False) or (x31 and False) or (x32 and True) or (x33 and False) or (x34 and True) or (x35 and False) or (x36 and False) or (x37 and True) or (x38 and False) or (x39 and False) or (x40 and True) or (x41 and False) or (x42 and False) or (x43 and True) or (x44 and False) or (x45 and True) or (x46 and True) or (x47 and True) or (x48 and True) or (x49 and True) or (x50 and True) or (x51 and True) or (x52 and False) or (x53 and True) or (x54 and True) or (x55 and True) or (x56 and True) or (x57 and True) or (x58 and False) or (x59 and True) or (x60 and True) or (x61 and True) or (x62 and True) or (x63 and False) or (x64 and True) or (x65 and True) or (x66 and True) or (x67 and True) or (x68 and True) or (x69 and True) or (x70 and False) or (x71 and False) or (x72 and True) or (x73 and False) or (x74 and False) or (x75 and False) or (x76 and False) or (x77 and False) or (x78 and False) or (x79 and True) or (x80 and True) or (x81 and False) or (x82 and False) or (x83 and True) or (x84 and False) or (x85 and True) or (x86 and True) or (x87 and True) or (x88 and True) or (x89 and False) or (x90 and True) or (x91 and False) or (x92 and True) or (x93 and False) or (x94 and True) or (x95 and True) or (x96 and True) or (x97 and False) or (x98 and False) or (x99 and True) or (x100 and True) or (x101 and False) or (x102 and True) or (x103 and False) or (x104 and True) or (x105 and True) or (x106 and False) or (x107 and True) or (x108 and True) or (x109 and False) or (x110 and True) or (x111 and False) or (x112 and True) or (x113 and True) or (x114 and False) or (x115 and False) or (x116 and True) or (x117 and True) or (x118 and True) or (x119 and False) or (x120 and True) or (x121 and True) or (x122 and False) or (x123 and True) or (x124 and True) or (x125 and False) or (x126 and True) or (x127 and False) or (x128 and False) or (x129 and True) or (x130 and True) or (x131 and False) or (x132 and False) or (x133 and False) or (x134 and True) or (x135 and True) or (x136 and False) or (x137 and True) or (x138 and True) or (x139 and False) or (x140 and True) or (x141 and True) or (x142 and False) or (x143 and True) or (x144 and False) or (x145 and True) or (x146 and True) or (x147 and False) or (x148 and False) or (x149 and False) or (x150 and False) or (x151 and False) or (x152 and False) or (x153 and True) or (x154 and False) or (x155 and False) or (x156 and False) or (x157 and False) or (x158 and False) or (x159 and False) or (x160 and False) or (x161 and False) or (x162 and True) or (x163 and False) or (x164 and True) or (x165 and False) or (x166 and True) or (x167 and True) or (x168 and False) or (x169 and False) or (x170 and True) or (x171 and True) or (x172 and True) or (x173 and False) or (x174 and True) or (x175 and False) or (x176 and False) or (x177 and False) or (x178 and False) or (x179 and False) or (x180 and False) or (x181 and True) or (x182 and False) or (x183 and False) or (x184 and False) or (x185 and False) or (x186 and False) or (x187 and False) or (x188 and False) or (x189 and False) or (x190 and False) or (x191 and True) or (x192 and True) or (x193 and True) or (x194 and True) or (x195 and True) or (x196 and False) or (x197 and True) or (x198 and False) or (x199 and False) or (x200 and False) or (x201 and False) or (x202 and True) or (x203 and False) or (x204 and False) or (x205 and False) or (x206 and False) or (x207 and False) or (x208 and False) or (x209 and False) or (x210 and False) or (x211 and False) or (x212 and False) or (x213 and False) or (x214 and False) or (x215 and False) or (x216 and False) or (x217 and False) or (x218 and False) or (x219 and False) or (x220 and False) or (x221 and False) or (x222 and False) or (x223 and False) or (x224 and False) or (x225 and False) or (x226 and True) or (x227 and True) or (x228 and False) or (x229 and True) or (x230 and False) or (x231 and False) or (x232 and False) or (x233 and False) or (x234 and False) or (x235 and False) or (x236 and False) or (x237 and False) or (x238 and True) or (x239 and False) or (x240 and False) or (x241 and False) or (x242 and True) or (x243 and False) or (x244 and True) or (x245 and True) or (x246 and False) or (x247 and True) or (x248 and False) or (x249 and False) or (x250 and True) or (x251 and False) or (x252 and False) or (x253 and False) or (x254 and False) or (x255 and True) or (x256 and False) or (x257 and True) or (x258 and True) or (x259 and False) or (x260 and False) or (x261 and True) or (x262 and False) or (x263 and False) or (x264 and False) or (x265 and False) or (x266 and False) or (x267 and False) or (x268 and False) or (x269 and False) or (x270 and False) or (x271 and False) or (x272 and False) or (x273 and False) or (x274 and True) or (x275 and False) or (x276 and True) or (x277 and False) or (x278 and True) or (x279 and False) or (x280 and False) or (x281 and True) or (x282 and False) or (x283 and False) or (x284 and False) or (x285 and True) or (x286 and False) or (x287 and False) or (x288 and False) or (x289 and False) or (x290 and False) or (x291 and False) or (x292 and False) or (x293 and False) or (x294 and False) or (x295 and False) or (x296 and False) or (x297 and False) or (x298 and False) or (x299 and True) or (x300 and True) or (x301 and False) or (x302 and True) or (x303 and False) or (x304 and False) or (x305 and True) or (x306 and False) or (x307 and False) or (x308 and False) or (x309 and True) or (x310 and False) or (x311 and True) or (x312 and True) or (x313 and False) or (x314 and True) or (x315 and True) or (x316 and False) or (x317 and False) or (x318 and False) or (x319 and False) or (x320 and True) or (x321 and False) or (x322 and False) or (x323 and False) or (x324 and False) or (x325 and False) or (x326 and False) or (x327 and False) or (x328 and True) or (x329 and False) or (x330 and True) or (x331 and True) or (x332 and True) or (x333 and True) or (x334 and True) or (x335 and True) or (x336 and False) or (x337 and True) or (x338 and False) or (x339 and True) or (x340 and True) or (x341 and True) or (x342 and True) or (x343 and True) or (x344 and False) or (x345 and False) or (x346 and False) or (x347 and False) or (x348 and False) or (x349 and False) or (x350 and False) or (x351 and False) or (x352 and False) or (x353 and False) or (x354 and False) or (x355 and False) or (x356 and False) or (x357 and True) or (x358 and False) or (x359 and False) or (x360 and False) or (x361 and True) or (x362 and True) or (x363 and False) or (x364 and True) or (x365 and True) or (x366 and True) or (x367 and True) or (x368 and True) or (x369 and True) or (x370 and False) or (x371 and False) or (x372 and False) or (x373 and False) or (x374 and False) or (x375 and False) or (x376 and False) or (x377 and False) or (x378 and True) or (x379 and False) or (x380 and False) or (x381 and False) or (x382 and False) or (x383 and False) or (x384 and False) or (x385 and False) or (x386 and False) or (x387 and False) or (x388 and True) or (x389 and False) or (x390 and False) or (x391 and True) or (x392 and True) or (x393 and True) or (x394 and False) or (x395 and True) or (x396 and False) or (x397 and False) or (x398 and False) or (x399 and False) or (x400 and False) or (x401 and False) or (x402 and False) or (x403 and False) or (x404 and False) or (x405 and False) or (x406 and False) or (x407 and False) or (x408 and False) or (x409 and False) or (x410 and False) or (x411 and False) or (x412 and False) or (x413 and False) or (x414 and True) or (x415 and False) or (x416 and False) or (x417 and True) or (x418 and False) or (x419 and False) or (x420 and True) or (x421 and True) or (x422 and False) or (x423 and False) or (x424 and True) or (x425 and False) or (x426 and False) or (x427 and False) or (x428 and F'
    
def test_mnist():
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")

    # Define training configuration.
    config = get_config()

    # Define the model.
    # soft = CNN()
    soft, _, _ = neural_logic_net.net(batch_nln)

    # Get the MNIST dataset.
    train_ds, test_ds = get_datasets()
    # If we're using a NLN then flatten the images
    train_ds["image"] = jnp.reshape(train_ds["image"], (train_ds["image"].shape[0], -1))
    test_ds["image"] = jnp.reshape(test_ds["image"], (test_ds["image"].shape[0], -1))
    
    # Train and evaluate the model.
    trained_state = train_and_evaluate(soft, (train_ds, test_ds), config=config, workdir="./mnist_metrics")

    # Check symbolic net
    _, hard, symbolic = neural_logic_net.net(nln)
    check_symbolic((soft, hard, symbolic), (train_ds, test_ds), trained_state)
