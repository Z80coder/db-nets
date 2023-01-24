import jax
import numpy
from plum import dispatch


@dispatch
def symbolic_operator(operator: str, x: str) -> str:
    return f'{operator}({x})'.replace('\'', '')


@dispatch
def symbolic_operator(operator: str, x: float, y: str):
    return symbolic_operator(operator, str(x), y)


@dispatch
def symbolic_operator(operator: str, x: str, y: float):
    return symbolic_operator(operator, x, str(y))


@dispatch
def symbolic_operator(operator: str, x: float, y: numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: numpy.ndarray, y: numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: str, y: str):
    return f'{operator}({x}, {y})'.replace('\'', '')


@dispatch
def symbolic_operator(operator: str, x: numpy.ndarray, y: float):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: list, y: float):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: list, y: list):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: bool, y: str):
    return symbolic_operator(operator, str(x), y)


@dispatch
def symbolic_operator(operator: str, x: str, y: numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: str, y: int):
    return symbolic_operator(operator, x, str(y))


@dispatch
def symbolic_operator(operator: str, x: tuple):
    return symbolic_operator(operator, str(x))


@dispatch
def symbolic_operator(operator: str, x: list, y: numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: numpy.ndarray, y: jax.numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x, y)


@dispatch
def symbolic_operator(operator: str, x: numpy.ndarray):
    return numpy.vectorize(symbolic_operator, otypes=[object])(operator, x)


@dispatch
def symbolic_operator(operator: str, x: list):
    return symbolic_operator(operator, numpy.array(x))

