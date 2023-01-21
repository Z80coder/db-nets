import jax

from neurallogic import neural_logic_net


def majority_index(input_size: int) -> int:
    return (input_size - 1) // 2


def soft_majority(x: jax.numpy.array) -> float:
    index = majority_index(x.shape[-1])
    sorted_x = jax.numpy.sort(x, axis=-1)
    return jax.numpy.take(sorted_x, index, axis=-1)


def hard_majority(x: jax.numpy.array) -> bool:
    threshold = x.shape[-1] - majority_index(x.shape[-1])
    return jax.numpy.sum(x, axis=-1) >= threshold


soft_majority_layer = jax.vmap(soft_majority, in_axes=0)

hard_majority_layer = jax.vmap(hard_majority, in_axes=0)


def symbolic_majority_layer(x):
    return hard_majority_layer(x)


majority_layer = neural_logic_net.select(
    soft_majority_layer, hard_majority_layer, symbolic_majority_layer)
