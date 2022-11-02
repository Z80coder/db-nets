import pytest
from neurallogic import hard_not

def test_differentiable_hard_not():
    assert hard_not.differentiable_hard_not(0.1, 0.6) == pytest.approx(0.42)