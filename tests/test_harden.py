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
    assert harden.harden([0.5, 0.6, 0.4, 0.0, 1.0]) == [
        False, True, False, False, True]


def test_harden_array():
    assert jnp.array_equal(harden.harden(
        jnp.array([0.5, 0.6, 0.4, 0.0, 1.0])), [False, True, False, False, True])


def test_harden_dict():
    dict = {'bit_a': 0.5, 'bit_b': 0.6, 'c': 0.4, 'bit_d': 0.0, 'bit_e': 1.0}
    expected_dict = {'bit_a': False, 'bit_b': True, 'c': 0.4, 'bit_d': False, 'bit_e': True}
    assert harden.harden(dict) == expected_dict


def test_harden_frozen_dict():
    dict = flax.core.frozen_dict.FrozenDict(
        {'a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0})
    expected_dict = {'a': 0.5, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0}
    assert harden.harden(dict) == expected_dict


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
    dict = {'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0}
    expected_dict = {'bit_a': False, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict(dict)
    assert harden.harden(dict) == expected_dict


def test_harden_compound_dict():
    dict = {'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0,
            'f': {'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0}}
    expected_dict = {'bit_a': False, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0, 'f': {
        'bit_a': False, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0}}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict(dict)
    assert harden.harden(dict) == expected_dict


def test_harden_complex_compound_dict():
    dict = {'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0, 'f': {
        'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'bit_d': 0.0, 'e': 1.0, 'g': [0.5, 0.6, 0.4, 0.0, 1.0]}}
    expected_dict = {'bit_a': False, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0, 'f': {
        'bit_a': False, 'bit_b': True, 'bit_c': False, 'bit_d': False, 'e': 1.0, 'g': [False, True, False, False, True]}}
    assert harden.harden(dict) == expected_dict
    dict = flax.core.frozen_dict.FrozenDict(dict)
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
        {'Soft_params': {'bit_a': 0.5, 'bit_b': 0.6, 'bit_c': 0.4, 'Soft_d': 0.0, 'e': 1.0}})
    expected_weights = flax.core.FrozenDict(
        {'Hard_params': {'bit_a': False, 'bit_b': True, 'bit_c': False, 'Hard_d': 0.0, 'e': 1.0}})
    hard_weights = harden.hard_weights(weights).unfreeze()
    assert str(hard_weights) == str(expected_weights.unfreeze())
