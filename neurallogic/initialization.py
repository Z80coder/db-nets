import jax


def initialize_uniform_range(lower=0.0, upper=1.0):
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        x = jax.random.uniform(key, shape, dtype, lower, upper)
        return x

    return init


def initialize_near_to_zero(mean=-1, std=0.5):
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # Sample from standard normal distribution (zero mean, unit variance)
        x = jax.random.normal(key, shape, dtype)
        # Transform to a normal distribution with mean -1 and standard deviation 0.5
        x = std * x + mean
        x = jax.numpy.clip(x, 0.001, 0.999)
        return x

    return init


def initialize_near_to_one():
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # Sample from standard normal distribution (zero mean, unit variance)
        x = jax.random.normal(key, shape, dtype)
        # Transform to a normal distribution with mean 1 and standard deviation 0.5
        x = 0.5 * x + 1
        x = jax.numpy.clip(x, 0.001, 0.999)
        return x

    return init


# TODO: get rid of symmetry
def initialize_bernoulli(p=0.5, low=0.001, high=0.999):
    def init(key, shape, dtype):
        x = jax.random.bernoulli(key, p, shape)
        x = jax.numpy.where(x, high, low)
        x = jax.numpy.asarray(x, dtype)
        return x

    return init

def initialize_bernoulli_uniform(p=0.5, low=0.001, high=0.999):
    def init(key, shape, dtype):
        x = jax.random.bernoulli(key, p, shape)
        h = jax.random.uniform(key, shape, dtype, 0.5, high)
        l = jax.random.uniform(key, shape, dtype, low, 0.5)
        x = jax.numpy.where(x, h, l)
        x = jax.numpy.asarray(x, dtype)
        return x

    return init
