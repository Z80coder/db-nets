from typing import Any, Callable, Optional, Tuple
import jax
from flax import linen as nn

def soft_not(w: float, x: float) -> float:
    """
    w > 0.5 implies the not operation is active, else inactive

    Assumes x is in [0, 1]
    
    Corresponding hard logic: ! (x XOR w)
    """
    w = jax.numpy.clip(w, 0.0, 1.0)
    return 1.0 - w + x * (2.0 * w - 1.0)

@jax.jit
def hard_not(w: bool, x: bool) -> bool:
    return ~(x ^ w)

soft_not_neuron = jax.vmap(soft_not, 0, 0)

hard_not_neuron = jax.vmap(hard_not, 0, 0)

soft_not_layer = jax.vmap(soft_not_neuron, (0, None), 0)

hard_not_layer = jax.vmap(hard_not_neuron, (0, None), 0)

class SoftNOT(nn.Module):
    """A NOT layer than transforms its inputs along the last dimension.

    Attributes:
        kernel_init: initializer function for the weight matrix.
        dtype: the dtype of the computation (default: infer from input and weights).
        weights_dtype: the dtype passed to the weight initializer (default: float32).
    """
    layer_size: int
    dtype: Optional[Any] = None
    weights_dtype: Any = jax.numpy.float32
    weights_init: Callable = nn.initializers.uniform(1.0)

    @nn.compact
    def __call__(self, x: Any) -> Any:
        weights = self.param('weights',
                        self.weights_init,
                        (self.layer_size, jax.numpy.shape(x)[-1]),
                        self.weights_dtype)
        x = jax.numpy.asarray(x, self.dtype)
        return soft_not_layer(weights, x)



