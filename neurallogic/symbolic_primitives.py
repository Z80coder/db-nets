import numpy
from plum import dispatch

@dispatch
def binary_operator(operator: str, a: str, b: str) -> str:
    return f"{a} {operator} {b}"

@dispatch
def binary_operator(operator: str, a: numpy.ndarray, b: numpy.ndarray):
    #print("binary_operator", operator, a, b)
    #print("element type", a.dtype, b.dtype)
    r = numpy.vectorize(binary_operator, otypes=[object])(operator, a, b)
    #print("r", r)
    #print("element type", r.dtype)
    return r

def symbolic_eval(x):
    return numpy.vectorize(eval)(x)


