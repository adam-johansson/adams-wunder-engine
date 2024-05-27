from GenHex_Simple import GenHex_single
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ----------------------------------Fluid inputs----------------------------------

t1_oil = 500
p1_oil = 10e5
m_oil = 6.6
h1_oil = PropsSI('Hmass', 'T', t1_oil, 'P', p1_oil, 'INCOMP::PNF2')

t1_h2 = 20
p1_h2 = 300e5
m_h2 = 0.080
h1_h2 = PropsSI('Hmass', 'T', t1_h2, 'P', p1_h2, 'Hydrogen')


def find_optimum(x):
    # Script for running
    f1 = type('f1', (), {})()
    f2 = type('f2', (), {})()

    HEX1 = type('HEX1', (), {})()

    # Engine oil
    f1.T01 = np.array([t1_oil])
    f1.p01 = np.array([p1_oil])
    f1.mdot = np.array([m_oil])
    f1.fluid = 'INCOMP::PNF2'  # some kind of heat transfer mineral oil
    f1.dataSource = 'Coolprop_incomp'

    # Hydrogen fuel
    f2.T01 = np.array([t1_h2])
    f2.p01 = np.array([p1_h2])
    f2.mdot = np.array([m_h2])
    f2.fluid = 'Hydrogen'
    f2.dataSource = 'coolprop'

    # Geometrical
    HEX1.wt = 0.0005
    HEX1.FinAR = 0.01 / (HEX1.wt ** 0.5)
    HEX1.Lx = np.array([0.2])
    HEX1.Ly = np.array([0.2])
    HEX1.Lz = np.array([0.2])

    HEX1.correlation = "KL"

    HEX1.f1_flowdir = 'Ly'
    HEX1.f1_L = 0.01  # dont change
    HEX1.f2_flowdir = 'Lx'
    HEX1.f2_L = 0.01  # dont change

    # Material
    HEX1.density = 2840  # Alu 2219
    HEX1.k = 120  # Conductivity # Alu 2219

    # ----------------------------------Generalized geometry input ----------------------------------
    HEX1.sigma_r = np.array([x[0]])
    HEX1.alpha_r = np.array([x[1]])
    HEX1.chi = np.array([x[2]])

    try:
        f1, f2, HEX1 = GenHex_single(f1, f2, HEX1, verbose=False)
    except ValueError:
        cost = 999


    #print('HEX effectiveness', HEX1.eps)

    HEX1.Q = np.reshape(HEX1.eps * HEX1.Cmin * abs(f1.T0[0] - f2.T0[0]), -1)

    t2_h2 = f2.T0[-1]
    p2_h2 = f2.p0[-1]
    dp_h2 = p1_h2 - p2_h2

    t2_oil = f1.T0[-1]
    p2_oil = f1.p0[-1]
    dp_oil = p1_oil - p2_oil

    cost = 1e3 * np.abs(t2_h2 - t2_goal) + dp_oil
    print(cost, x[0], x[1], x[2], dp_oil, dp_h2, t2_oil, HEX1.Q*1e-3, t2_h2)

    return cost


sigma_limits = (0.5, 2.0)
alpha_limits = (0.5, 2.0)
chi_limits = (0.06, 0.3)

limits = [sigma_limits, alpha_limits, chi_limits]

t2_goal = 450

x0 = np.array([0.5, 0.5, 0.09])

sol = minimize(find_optimum, x0, bounds=limits)





