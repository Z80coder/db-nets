from neurallogic import hard_masks
from tests import utils


def test_mask_to_true():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 1.0],
        [[0.0, 1.0], 1.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 1.0],
        [[-0.1, 1.0], 1.0],
    ]
    for input, expected in test_data:
        utils.check_consistency(
            hard_masks.soft_mask_to_true,
            hard_masks.hard_mask_to_true,
            expected,
            input[0],
            input[1],
        )


def test_mask_to_false():
    test_data = [
        [[1.0, 1.0], 1.0],
        [[1.0, 0.0], 0.0],
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 0.0],
        [[1.1, 1.0], 1.0],
        [[1.1, 0.0], 0.0],
        [[-0.1, 0.0], 0.0],
        [[-0.1, 1.0], 0.0],
    ]
    for input, expected in test_data:
        utils.check_consistency(
            hard_masks.soft_mask_to_false,
            hard_masks.hard_mask_to_false,
            expected,
            input[0],
            input[1],
        )
