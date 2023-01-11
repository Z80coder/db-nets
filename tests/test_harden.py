import jax.numpy as jnp
from neurallogic import harden
import flax


def test_harden_float():
    assert harden.harden_float(0.5) == False
    assert harden.harden_float(0.6) == True
    assert harden.harden_float(0.4) == False
    assert harden.harden_float(0.0) == False
    assert harden.harden_float(1.0) == True


def test_harden_list():
    assert harden.harden_list([0.5, 0.6, 0.4, 0.0, 1.0]) == [
        False, True, False, False, True]


def test_harden_array():
    assert jnp.array_equal(harden.harden_array(
        jnp.array([0.5, 0.6, 0.4, 0.0, 1.0])), [False, True, False, False, True])


def test_harden_dict():
    dict = {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0}
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True}
    assert harden.harden_dict(dict) == expected_dict


def test_harden_frozen_dict():
    dict = flax.core.frozen_dict.FrozenDict(
        {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0})
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True}
    assert harden.harden_dict(dict) == expected_dict


def test_harden():
    assert harden.harden(0.5) == False
    assert harden.harden(0.6) == True
    assert harden.harden(0.4) == False
    assert harden.harden(0.0) == False
    assert harden.harden(1.0) == True
    assert harden.harden([0.5, 0.6, 0.4, 0.0, 1.0]) == [
        False, True, False, False, True]
    assert jnp.array_equal(harden.harden(jnp.array([0.5, 0.6, 0.4, 0.0, 1.0])), [
                           False, True, False, False, True])
    dict = {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0}
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict(
        {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0})
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True}
    assert harden.harden(dict) == expected_dict


def test_harden_compound_dict():
    dict = {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0,
            'f': {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0}}
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'f': {
        'a': False, 'b': True, 'c': False, 'd': False, 'e': True}}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict({'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0, 'f': {
                                            'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0}})
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'f': {
        'a': False, 'b': True, 'c': False, 'd': False, 'e': True}}
    assert harden.harden(dict) == expected_dict


def test_harden_complex_compound_dict():
    dict = {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0, 'f': {
        'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0, 'g': [0.5, 0.6, 0.4, 0.0, 1.0]}}
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'f': {
        'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'g': [False, True, False, False, True]}}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict({'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0, 'f': {
                                            'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0, 'e': 1.0, 'g': [0.5, 0.6, 0.4, 0.0, 1.0]}})
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'f': {
        'a': False, 'b': True, 'c': False, 'd': False, 'e': True, 'g': [False, True, False, False, True]}}
    assert harden.harden(dict) == expected_dict


def test_dict_with_array():
    dict = {'a': 0.5, 'b': 0.6, 'c': 0.4, 'd': 0.0,
            'e': 1.0, 'f': jnp.array([0.5, 0.6, 0.4, 0.0, 1.0])}
    expected_dict = {'a': False, 'b': True, 'c': False, 'd': False,
                     'e': True, 'f': jnp.array([False, True, False, False, True])}
    str(harden.harden(dict)) == str(expected_dict)


def test_harden_compound_list():
    list = [0.5, 0.6, 0.4, 0.0, 1.0, [0.5, 0.6, 0.4, 0.0, 1.0]]
    expected_list = [False, True, False, False,
                     True, [False, True, False, False, True]]
    assert harden.harden(list) == expected_list


def test_hard_weights():
    weights = flax.core.FrozenDict(
        {'Soft_params': {'a': 0.5, 'b': 0.6, 'c': 0.4, 'Soft_d': 0.0, 'e': 1.0}})
    expected_weights = flax.core.FrozenDict(
        {'Hard_params': {'a': False, 'b': True, 'c': False, 'Hard_d': False, 'e': True}})
    hard_weights = harden.hard_weights(weights).unfreeze()
    assert str(hard_weights) == str(expected_weights.unfreeze())
