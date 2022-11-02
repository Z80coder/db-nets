def differentiable_hard_not(x, w):
    """
    w > 0.5 implies the NOT operation is active
    else the NOT operation is inactive
    The corresponding hard logic is: (x AND w) || (! x AND ! w) or equivalently ! (x XOR w)
    """
    return [1.0 - w[i] + x[i] * (2.0 * w[i] - 1.0) for i in range(len(x))]
