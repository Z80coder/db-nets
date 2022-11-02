from neurallogic import differentiable_hard_not

def test_differentiable_hard_not():
    assert differentiable_hard_not([0.1, 0.5], [0.8, 0.3]) == [0.0, 0.0]