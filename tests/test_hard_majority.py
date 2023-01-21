import numpy
import jax

from neurallogic import hard_majority, harden, symbolic_generation


def test_majority_index():
    assert hard_majority.majority_index(1) == 0
    assert hard_majority.majority_index(2) == 0
    assert hard_majority.majority_index(3) == 1
    assert hard_majority.majority_index(4) == 1
    assert hard_majority.majority_index(5) == 2
    assert hard_majority.majority_index(6) == 2
    assert hard_majority.majority_index(7) == 3
    assert hard_majority.majority_index(8) == 3
    assert hard_majority.majority_index(9) == 4
    assert hard_majority.majority_index(10) == 4
    assert hard_majority.majority_index(11) == 5
    assert hard_majority.majority_index(12) == 5


def test_soft_majority():
    assert hard_majority.soft_majority(numpy.array([1.0])) == 1.0
    assert hard_majority.soft_majority(numpy.array([2.0, 1.0])) == 1.0
    assert hard_majority.soft_majority(numpy.array([1.0, 3.0, 2.0])) == 2.0
    assert hard_majority.soft_majority(
        numpy.array([2.0, 1.0, 4.0, 3.0])) == 2.0
    assert hard_majority.soft_majority(
        numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 3.0
    assert hard_majority.soft_majority(
        numpy.array([6.0, 3.0, 2.0, 4.0, 5.0, 1.0])) == 3.0
    assert hard_majority.soft_majority(numpy.array(
        [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])) == 4.0
    assert hard_majority.soft_majority(numpy.array(
        [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0])) == 4.0
    assert hard_majority.soft_majority(numpy.array(
        [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 9.0, 8.0])) == 5.0
    assert hard_majority.soft_majority(numpy.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])) == 5.0
    assert hard_majority.soft_majority(numpy.array(
        [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])) == 6.0
    assert hard_majority.soft_majority(numpy.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])) == 6.0


def test_hard_majority():
    assert hard_majority.hard_majority(numpy.array([True])) == True
    assert hard_majority.hard_majority(numpy.array([False])) == False
    assert hard_majority.hard_majority(numpy.array([True, False])) == False
    assert hard_majority.hard_majority(
        numpy.array([False, True, False])) == False
    assert hard_majority.hard_majority(
        numpy.array([True, False, True, False])) == False
    assert hard_majority.hard_majority(numpy.array(
        [False, True, False, True, False])) == False
    assert hard_majority.hard_majority(numpy.array(
        [True, True, True, False, True, False])) == True
    assert hard_majority.hard_majority(numpy.array(
        [True, False, False, True, True, True, False])) == True
    assert hard_majority.hard_majority(numpy.array(
        [False, True, False, True, False, True, False, True])) == False
    assert hard_majority.hard_majority(numpy.array(
        [True, True, True, True, True, False, True, True, True])) == True
    assert hard_majority.hard_majority(numpy.array(
        [True, False, False, False, False, False, True, True, True, True])) == False


def test_soft_and_hard_majority_equivalence():
    soft_maj = jax.jit(hard_majority.soft_majority)
    hard_maj = jax.jit(hard_majority.hard_majority)
    for i in range(1, 50):
        input = numpy.random.rand(i)
        soft_output = soft_maj(input)
        hard_output = hard_maj(harden.harden(input))
        assert harden.harden(soft_output) == hard_output


def test_soft_majority_layer():
    assert numpy.all(hard_majority.soft_majority_layer(
        numpy.array([[2.0, 1.0], [1.0, 2.0]])) == numpy.array([1.0, 1.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])) == numpy.array([2.0, 2.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])) == numpy.array([2.0, 2.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([3.0, 3.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([3.0, 3.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([4.0, 4.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([4.0, 4.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([5.0, 5.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([5.0, 5.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([6.0, 6.0]))
    assert numpy.all(hard_majority.soft_majority_layer(numpy.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])) == numpy.array([6.0, 6.0]))


def test_hard_majority_layer():
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array(
        [[True, False], [False, True]])) == numpy.array([False, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array(
        [[True, False, True], [True, False, False]])) == numpy.array([True, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array(
        [[True, False, True, False], [False, True, False, True]])) == numpy.array([False, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array(
        [[True, False, True, False, True], [True, False, True, False, True]])) == numpy.array([True, True]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True, False, True, False], [
                     False, True, False, True, False, True]])) == numpy.array([False, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True, False, True, False, True], [
                     True, False, True, False, True, False, False]])) == numpy.array([True, False]))

    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False], [
                     False, True], [False, True]])) == numpy.array([False, False, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True], [
                     True, False, True], [True, False, True]])) == numpy.array([True, True, True]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True, False], [
                     False, True, False, True], [False, True, False, True]])) == numpy.array([False, False, False]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True, False, True], [
                     True, False, True, False, True], [True, False, True, False, True]])) == numpy.array([True, True, True]))
    assert numpy.all(hard_majority.hard_majority_layer(numpy.array([[True, False, True, False, True, False], [
                     False, True, False, True, False, True], [False, True, False, True, False, True]])) == numpy.array([False, False, False]))


def test_majority_layer():
    soft, hard, symbolic = hard_majority.soft_majority_layer, hard_majority.hard_majority_layer, hard_majority.symbolic_majority_layer

    test_data = [
        [
            [[0.8, 0.1, 0.4], [1.0, 0.0, 0.3]],
            [0.4, 0.3]
        ],
        [
            [[0.8, 0.1, 0.4], [1.0, 0.0, 0.3], [0.0, 0.0, 0.0]],
            [0.4, 0.3, 0.0]
        ],
        [
            [[0.8, 0.1, 0.4], [1.0, 0.0, 0.3], [0.8, 0.9, 0.1], [0.2, 0.01, 0.45]],
            [0.4, 0.3, 0.8, 0.2]
        ],
        [
            [[0.8, 0.1, 0.4], [1.0, 0.0, 0.3], [0.8, 0.9, 0.1], [0.2, 0.01, 0.45], [0.0, 0.0, 0.0]],
            [0.4, 0.3, 0.8, 0.2, 0.0]
        ],
        [
            [[0.3, 0.93, 0.01, 0.5], [0.2, 0.01, 0.45, 0.1], [0.8, 0.9, 0.1, 0.2], [0.8, 0.1, 0.4, 0.3], [0.0, 0.0, 0.0, 0.0]],
            [0.3, 0.1, 0.2, 0.3, 0.0]
        ]
    ]
    
    for input, expected in test_data:
        input = jax.numpy.array(input)
        expected = jax.numpy.array(expected)
        soft_output = soft(input)
        assert jax.numpy.array_equal(soft_output, expected)
        hard_output = hard(harden.harden(input))
        assert jax.numpy.array_equal(hard_output, harden.harden(expected))
        jaxpr = symbolic_generation.make_symbolic_jaxpr(symbolic, harden.harden(input))
        symbolic_output = symbolic_generation.symbolic_expression(jaxpr, harden.harden(input))
        assert jax.numpy.array_equal(symbolic_output, harden.harden(expected))

# TODO: test training the hard majority layer