import numpy as np
from timeit import default_timer as timer

from piston_engine.src.piston import valve_isentrop, walls, wiebe, port_isentrop

# from piston_engine.src.piston import thermo_computations, polynomials
from piston_engine.src.misc import post_processing
from piston_engine.src.misc.entropy import entropy_calc

# from piston_engine.src.piston.fuel_func import fuel_props
# from piston_engine.src.piston.fluid_props import properties

import thermo

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


from numba import njit


@njit()
def dxdphi(t, x):
    # solve a system of ODEs for pressure, temperature, volume
    # assign ode to vector element

    X = x[0]  # temperature
    v = x[1]  # volume
    a = x[2]

    g = 9.81
    cd = 0.5

    # Zone 1 (unburned mixture) conservation equations

    # mass = + fuel - flame front - from zone 1 to 2
    dm1 = dm_f - dm_ff - dm_12

    dsigma_i1 = dsigma_if - dsigma_iff - dsigma_i12

    dU1 = dm_f * h_f - dm_ff * h_1f - dm_12 * h_12 + dQ_w1 - pdV1

    # Zone 2 (burned mixture) conservation equations

    # mass = mass from flame front and mass directly from 1
    dm2 = dmff + dm12

    # idk what this is
    dsigmai2 = dsigmaiF + dsigmai12

    hf2 = h1f + deltahv

    dQb = deltahv_f * dm_f + m2 * deltah_v2

    # Internal energy
    dU2 = dmff * hf2 + dm12 * h12 + dQw2 - pdV2

    return [dxdt, dvdt, dadt]


x0 = 0
v0 = 0
a0 = 0

t = np.linspace(0, 10)

# Init simulation
x0 = [x0, v0, a0]  # initial state value


sol = solve_ivp(
    dxdphi,
    t_span=(min(t), max(t)),
    method="LSODA",
    y0=x0,
    t_eval=t,
    rtol=1e-10,
    atol=1e-12,
)

plt.plot(t, sol.y[1])
plt.show()
