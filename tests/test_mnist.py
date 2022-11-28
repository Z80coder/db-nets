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
    x = hard_or.or_layer(type)(100, nn.initializers.uniform(1.0), dtype=jnp.float32)(x) # >=1700 need for >98% accuracy
    x = hard_not.not_layer(type)(10, dtype=jnp.float32)(x)
    x = primitives.nl_ravel(type)(x) # flatten the outputs of the not layer
    x = harden_layer.harden_layer(type)(x) # harden the outputs of the not layer
    x = primitives.nl_reshape(type)((10, 100))(x) # reshape to 10 ports, 100 bits each
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
    print("symbolic_output", symbolic_output[0][:10000])
    assert symbolic_output[0][:10000] == '((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((not(((x0 and True) or (x1 and False) or (x2 and False) or (x3 and False) or (x4 and True) or (x5 and True) or (x6 and True) or (x7 and True) or (x8 and False) or (x9 and False) or (x10 and False) or (x11 and True) or (x12 and True) or (x13 and False) or (x14 and False) or (x15 and False) or (x16 and True) or (x17 and True) or (x18 and False) or (x19 and True) or (x20 and True) or (x21 and False) or (x22 and True) or (x23 and True) or (x24 and False) or (x25 and False) or (x26 and True) or (x27 and True) or (x28 and True) or (x29 and True) or (x30 and True) or (x31 and True) or (x32 and False) or (x33 and False) or (x34 and True) or (x35 and True) or (x36 and True) or (x37 and True) or (x38 and True) or (x39 and False) or (x40 and False) or (x41 and False) or (x42 and True) or (x43 and True) or (x44 and True) or (x45 and False) or (x46 and False) or (x47 and False) or (x48 and True) or (x49 and False) or (x50 and False) or (x51 and False) or (x52 and True) or (x53 and True) or (x54 and False) or (x55 and True) or (x56 and False) or (x57 and False) or (x58 and True) or (x59 and True) or (x60 and True) or (x61 and False) or (x62 and False) or (x63 and False) or (x64 and True) or (x65 and False) or (x66 and False) or (x67 and False) or (x68 and False) or (x69 and True) or (x70 and True) or (x71 and True) or (x72 and False) or (x73 and False) or (x74 and False) or (x75 and True) or (x76 and True) or (x77 and False) or (x78 and True) or (x79 and True) or (x80 and True) or (x81 and True) or (x82 and False) or (x83 and True) or (x84 and True) or (x85 and True) or (x86 and False) or (x87 and False) or (x88 and True) or (x89 and False) or (x90 and True) or (x91 and False) or (x92 and False) or (x93 and True) or (x94 and False) or (x95 and False) or (x96 and False) or (x97 and False) or (x98 and False) or (x99 and False) or (x100 and False) or (x101 and False) or (x102 and False) or (x103 and True) or (x104 and True) or (x105 and True) or (x106 and True) or (x107 and True) or (x108 and False) or (x109 and False) or (x110 and True) or (x111 and False) or (x112 and False) or (x113 and False) or (x114 and False) or (x115 and True) or (x116 and False) or (x117 and False) or (x118 and True) or (x119 and False) or (x120 and False) or (x121 and False) or (x122 and False) or (x123 and False) or (x124 and False) or (x125 and False) or (x126 and False) or (x127 and False) or (x128 and False) or (x129 and False) or (x130 and False) or (x131 and False) or (x132 and False) or (x133 and False) or (x134 and False) or (x135 and False) or (x136 and False) or (x137 and False) or (x138 and False) or (x139 and True) or (x140 and False) or (x141 and True) or (x142 and False) or (x143 and False) or (x144 and False) or (x145 and True) or (x146 and True) or (x147 and False) or (x148 and False) or (x149 and False) or (x150 and False) or (x151 and False) or (x152 and False) or (x153 and False) or (x154 and False) or (x155 and False) or (x156 and False) or (x157 and False) or (x158 and False) or (x159 and False) or (x160 and False) or (x161 and False) or (x162 and False) or (x163 and False) or (x164 and False) or (x165 and False) or (x166 and False) or (x167 and False) or (x168 and True) or (x169 and True) or (x170 and False) or (x171 and True) or (x172 and False) or (x173 and True) or (x174 and True) or (x175 and False) or (x176 and False) or (x177 and False) or (x178 and False) or (x179 and False) or (x180 and False) or (x181 and False) or (x182 and False) or (x183 and False) or (x184 and False) or (x185 and False) or (x186 and False) or (x187 and False) or (x188 and False) or (x189 and False) or (x190 and False) or (x191 and False) or (x192 and False) or (x193 and False) or (x194 and False) or (x195 and False) or (x196 and True) or (x197 and True) or (x198 and True) or (x199 and False) or (x200 and False) or (x201 and True) or (x202 and False) or (x203 and False) or (x204 and False) or (x205 and False) or (x206 and False) or (x207 and False) or (x208 and False) or (x209 and False) or (x210 and False) or (x211 and False) or (x212 and False) or (x213 and False) or (x214 and False) or (x215 and False) or (x216 and False) or (x217 and False) or (x218 and False) or (x219 and False) or (x220 and False) or (x221 and False) or (x222 and False) or (x223 and False) or (x224 and True) or (x225 and False) or (x226 and True) or (x227 and False) or (x228 and True) or (x229 and False) or (x230 and False) or (x231 and False) or (x232 and False) or (x233 and False) or (x234 and False) or (x235 and False) or (x236 and False) or (x237 and False) or (x238 and False) or (x239 and False) or (x240 and False) or (x241 and False) or (x242 and False) or (x243 and False) or (x244 and False) or (x245 and False) or (x246 and False) or (x247 and False) or (x248 and False) or (x249 and False) or (x250 and False) or (x251 and True) or (x252 and False) or (x253 and False) or (x254 and False) or (x255 and True) or (x256 and False) or (x257 and True) or (x258 and False) or (x259 and False) or (x260 and False) or (x261 and False) or (x262 and False) or (x263 and False) or (x264 and False) or (x265 and False) or (x266 and False) or (x267 and False) or (x268 and False) or (x269 and False) or (x270 and False) or (x271 and False) or (x272 and False) or (x273 and False) or (x274 and False) or (x275 and False) or (x276 and False) or (x277 and False) or (x278 and False) or (x279 and False) or (x280 and False) or (x281 and True) or (x282 and True) or (x283 and False) or (x284 and True) or (x285 and False) or (x286 and False) or (x287 and False) or (x288 and False) or (x289 and False) or (x290 and False) or (x291 and False) or (x292 and False) or (x293 and False) or (x294 and False) or (x295 and False) or (x296 and False) or (x297 and False) or (x298 and False) or (x299 and False) or (x300 and False) or (x301 and False) or (x302 and False) or (x303 and False) or (x304 and False) or (x305 and False) or (x306 and False) or (x307 and False) or (x308 and False) or (x309 and False) or (x310 and False) or (x311 and False) or (x312 and True) or (x313 and False) or (x314 and False) or (x315 and False) or (x316 and False) or (x317 and False) or (x318 and False) or (x319 and False) or (x320 and False) or (x321 and False) or (x322 and False) or (x323 and True) or (x324 and False) or (x325 and False) or (x326 and False) or (x327 and False) or (x328 and False) or (x329 and False) or (x330 and False) or (x331 and False) or (x332 and False) or (x333 and True) or (x334 and True) or (x335 and True) or (x336 and False) or (x337 and True) or (x338 and True) or (x339 and False) or (x340 and False) or (x341 and False) or (x342 and False) or (x343 and False) or (x344 and False) or (x345 and False) or (x346 and False) or (x347 and False) or (x348 and False) or (x349 and False) or (x350 and False) or (x351 and False) or (x352 and True) or (x353 and False) or (x354 and False) or (x355 and False) or (x356 and False) or (x357 and False) or (x358 and False) or (x359 and False) or (x360 and False) or (x361 and False) or (x362 and True) or (x363 and True) or (x364 and True) or (x365 and False) or (x366 and True) or (x367 and True) or (x368 and False) or (x369 and False) or (x370 and False) or (x371 and False) or (x372 and False) or (x373 and False) or (x374 and False) or (x375 and False) or (x376 and False) or (x377 and False) or (x378 and False) or (x379 and False) or (x380 and False) or (x381 and True) or (x382 and False) or (x383 and False) or (x384 and False) or (x385 and False) or (x386 and False) or (x387 and False) or (x388 and False) or (x389 and False) or (x390 and False) or (x391 and False) or (x392 and True) or (x393 and True) or (x394 and False) or (x395 and False) or (x396 and False) or (x397 and False) or (x398 and False) or (x399 and False) or (x400 and False) or (x401 and False) or (x402 and False) or (x403 and False) or (x404 and False) or (x405 and False) or (x406 and False) or (x407 and False) or (x408 and False) or (x409 and False) or (x410 and False) or (x411 and False) or (x412 and False) or (x413 and False) or (x414 and False) or (x415 and False) or (x416 and False) or (x417 and False) or (x418 and True) or (x419 and False) or (x420 and True) or (x421 and True) or (x422 and False) or (x423 and False) or (x424 and False) or (x425 and False) or (x426 and False) or (x427 and False) or (x428 and False) or (x429 and False) or (x430 and False) or (x431 and False) or (x432 and False) or (x433 and False) or (x434 and False) or (x435 and False) or (x436 and False) or (x437 and False) or (x438 and False) or (x439 and False) or (x440 and False) or (x441 and False) or (x442 and False) or (x443 and False) or (x444 and False) or (x445 and False) or (x446 and True) or (x447 and False) or (x448 and False) or (x449 and False) or (x450 and False) or (x451 and False) or (x452 and False) or (x453 and False) or (x454 and False) or (x455 and False) or (x456 and False) or (x457 and False) or (x458 and False) or (x459 and False) or (x460 and False) or (x461 and False) or (x462 and False) or (x463 and False) or (x464 and False) or (x465 and False) or (x466 and False) or (x467 and False) or (x468 and False) or (x469 and False) or (x470 and False) or (x471 and False) or (x472 and False) or (x473 and False) or (x474 and True) or (x475 and False) or (x476 and False) or (x477 and False) or (x478 and True) or (x479 and True) or (x480 and False) or (x481 and False) or (x482 and False) or (x483 and False) or (x484 and False) or (x485 and False) or (x486 and False) or (x487 and False) or (x488 and False) or (x489 and False) or (x490 and False) or (x491 and False) or (x492 and False) or (x493 and False) or (x494 and False) or (x495 and False) or (x496 and False) or (x497 and False) or (x498 and False) or (x499 and False) or (x500 and False) or (x501 and False) or (x502 and True) or (x503 and True) or (x504 and False) or (x505 and '
    
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
