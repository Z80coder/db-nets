import numpy
from plum import dispatch

@dispatch
def binary_operator(operator: str, a: str, b: str) -> str:
    return f"({a} {operator} {b})"

@dispatch
def binary_operator(operator: str, a: numpy.ndarray, b: numpy.ndarray):
    return numpy.vectorize(binary_operator)(operator, a, b)

def symbolic_eval(x):
    return numpy.vectorize(eval)(x)


