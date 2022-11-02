def differentiable_hard_not(input, weights):
    return 1 - weights + input * (2 * weights - 1)