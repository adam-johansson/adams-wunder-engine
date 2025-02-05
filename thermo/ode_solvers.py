import numpy as np

def euler(f, y0, t, x0):
    """
    Euler method to solve ODEs.

    Parameters:
    f : function
        The function defining the ODE dy/dt = f(t, y).
    y0 : float
        The initial condition.
    t : array-like
        The time points where the solution is computed.

    Returns:
    y : array
        The solution of the ODE at the given time points.
    """
    n = len(t)
    y = np.zeros(n)
    dydt_array = np.zeros(n)
    y[0] = y0

    for i in range(1, n):
        print(f"Step {i} out of {n}")
        h = t[i] - t[i - 1]
        dydt, x0 = f(t[i - 1], y[i - 1], x0)
        y[i] = y[i - 1] + h * dydt
        dydt_array[i-1] = dydt

    return y, dydt_array


def euler_cantera(f, y0, t):
    """
    Euler method to solve ODEs.

    Parameters:
    f : function
        The function defining the ODE dy/dt = f(t, y).
    y0 : float
        The initial condition.
    t : array-like
        The time points where the solution is computed.

    Returns:
    y : array
        The solution of the ODE at the given time points.
    """
    n = len(t)
    y = np.zeros(n)
    dydt_array = np.zeros(n)
    y[0] = y0

    for i in range(1, n):
        print(f"Step {i} out of {n}")
        h = t[i] - t[i - 1]
        dydt = f(t[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * dydt
        dydt_array[i-1] = dydt

    return y, dydt_array


def runge_kutta(f, y0, t0, tf, h):
    """
    Solve the ODE y' = f(t, y) using the 4th order Runge-Kutta method.

    Parameters:
    f  : function
        The function f(t, y) that defines the ODE.
    y0 : float
        The initial value of y at t0.
    t0 : float
        The initial time.
    tf : float
        The final time.
    h  : float
        The step size.

    Returns:
    t_values : list
        List of time points.
    y_values : list
        List of y values corresponding to the time points.
    """
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0

    while t < tf:
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h

        t_values.append(t)
        y_values.append(y)

    return t_values, y_values


