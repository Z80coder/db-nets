import typing

import jax
import numpy
from plum import dispatch


@dispatch
def map_at_elements(x: str, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: bool, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: numpy.bool_, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: float, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: numpy.float32, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: numpy.int32, func: typing.Callable):
    return func(x)


@dispatch
def map_at_elements(x: list, func: typing.Callable):
    return [map_at_elements(item, func) for item in x]


@dispatch
def map_at_elements(x: numpy.ndarray, func: typing.Callable):
    return numpy.array([map_at_elements(item, func) for item in x], dtype=object)


@dispatch
def map_at_elements(x: jax.numpy.ndarray, func: typing.Callable):
    if x.ndim == 0:
        return func(x.item())
    return jax.numpy.array([map_at_elements(item, func) for item in x])


@dispatch
def map_at_elements(x: dict, func: typing.Callable):
    return {k: map_at_elements(v, func) for k, v in x.items()}


@dispatch
def map_at_elements(x: tuple, func: typing.Callable):
    return tuple(map_at_elements(list(x), func))
