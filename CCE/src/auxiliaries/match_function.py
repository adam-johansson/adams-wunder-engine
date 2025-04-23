import numpy as np


def fmatch(func, bound1, bound2, tol):
    x = (bound1 + bound2) / 2  # initial guess in the middle
    res = 9999
    x_low = bound1
    x_high = bound2

    if np.sign(func(bound1)) == np.sign(func(bound2)):
        raise Exception("No solution within given boundary")

    while abs(res) > tol:
        res = func(x)
        if res > 0:
            x_high = x  # higher boundary now previous value
            x = (x + x_low) / 2  # decrease input if output too high
        else:
            x_low = x  # lower boundary now previous value
            x = (x + x_high) / 2  # increase input if output too low

    return x
