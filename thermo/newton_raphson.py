import matplotlib.pyplot as plt
import numpy as np
from numba import njit


# @jit(nopython=True, cache=True)
def newton_method(fun, x0, jacobian):

    # initial guess
    x_n = np.atleast_2d(x0).T

    # tolerance for convergence
    tol = 1e-7

    # step length
    alpha = 1e-3

    # maximum number of iterations
    max_iter = int(1e5)

    # to track the residuals
    errors = np.zeros([10, max_iter])

    # get scaling factors
    # scale = 1 / np.abs(fun(x_n))

    # fast
    scale = np.atleast_2d([1, 1, 1e-3, 1e-3, 1e-3, 1e-5, 1, 1e2, 1, 1]).T

    # slow
    # scale = np.atleast_2d([1, 1e2, 1e2, 1e-3, 1e-1, 1e-5, 1e1, 1e3, 1, 1e1]).T
    for i in range(max_iter):
        F = fun(x_n)
        F_scaled = F * scale
        J = jacobian(x_n)
        J_scaled = J * scale

        errors[:, i] = F[:, 0]

        # print(f"Mean error: {np.mean(np.abs(F))}")
        # if np.mean(np.abs(F_scaled)) < tol:
        if np.max(np.abs(F_scaled)) < tol:
            print(f"Converged after {i} iterations.")
            break

        dx = -np.linalg.solve(J_scaled, F_scaled)
        x_n = x_n + dx * alpha

    """
    plt.plot(errors[0, : i], label="Eq 1", linewidth=2)
    plt.plot(errors[1, : i], label="Eq 2")
    plt.plot(errors[2, : i], label="Eq 3")
    plt.plot(errors[3, : i], label="Eq 4")
    plt.plot(errors[4, : i], label="Eq 5")
    plt.plot(errors[5, : i], label="Eq 6", linewidth=2)
    plt.plot(errors[6, : i], label="Eq 7")
    plt.plot(errors[7, : i], label="Eq 8", linewidth=3)
    plt.plot(errors[8, : i], label="Eq 9")
    plt.plot(errors[9, : i], label="Eq 10")
    plt.legend()
    plt.show()
    """

    return x_n.T
