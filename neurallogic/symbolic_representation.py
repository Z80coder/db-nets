import numpy
from plum import dispatch


@dispatch
def symbolic_representation(x: numpy.ndarray):
    return repr(x).replace('array', 'numpy.array').replace('\n', '').replace('float32', 'numpy.float32').replace('\'', '')


@dispatch
def symbolic_representation(x: str):
    return x.replace('\'', '')
