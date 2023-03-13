import numpy
import jax

from neurallogic import hard_count

def test_soft_count():
    # 2 bits are high in a 7-bit input array, x
    x = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    y = hard_count.soft_count(x)
    # We expect a 8-bit output array, y, where y[5] is the only high soft-bit (indicating that 5 soft-bits are low in the input)
    expected_output = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

    # Same example as above, except instead of 0s and 1s, we have soft-bits
    x = numpy.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])
    y = hard_count.soft_count(x)
    # We expect an 8-bit output array, y, where y[5] is the only high soft-bit (indicating that 5 soft-bits are low in the input)
    expected_output = numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.10000002, 0.10000002])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

    x = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = hard_count.soft_count(x)
    # We expect an 8-bit output array, y, where no y[0] is high (indicating that 0 soft-bits are low in the input)
    expected_output = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

    # Same example as above, except instead of 0s and 1s, we have soft-bits
    x = numpy.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    y = hard_count.soft_count(x)
    # We expect an 8-bit output array, y, where y[0] is high (indicating that 0 soft-bits are low in the input)
    expected_output = numpy.array([0.9, 0.10000002, 0.10000002, 0.10000002, 0.10000002, 0.10000002, 0.10000002, 0.10000002])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

    x = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = hard_count.soft_count(x)
    # We expect an 8-bit output array, y, where y[7] is the only high soft-bit (indicating that 7 soft-bits are low in the input)
    expected_output = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

    # Same example as above, except instead of 0s and 1s, we have soft-bits
    x = numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    y = hard_count.soft_count(x)
    # We expect an 7-bit output array, y, where y[7] is the only high soft-bit (indicating that 7 soft-bits are low in the input)
    expected_output = numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    print("soft_count", y)
    assert numpy.allclose(y, expected_output)

# TODO: test soft_count == hard_count