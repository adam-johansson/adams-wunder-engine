from GenHex_Simple import GenHex_single
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, basinhopping


# ----------------------------------Fluid inputs----------------------------------

t1_oil = 475  # oil temperature after oil-h2 hex
p1_oil = 10e5  # almost no pressure loss in oil-h2 hex
m_oil = 6.6
h1_oil = PropsSI('Hmass', 'T', t1_oil, 'P', p1_oil, 'INCOMP::PNF2')

t1_air = 265
p1_air = 47e3  # 0.47 bar
m_air = 20  # this is the flow from the bypass air that goes through the hx
h1_air = PropsSI('Hmass', 'T', t1_air, 'P', p1_air, 'Air')


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

    # Bypass air
    f2.T01 = np.array([t1_air])
    f2.p01 = np.array([p1_air])
    f2.mdot = np.array([m_air])
    f2.fluid = 'Air'
    f2.dataSource = 'coolprop'

    # Geometrical
    HEX1.wt = 0.0005
    HEX1.FinAR = 0.01 / (HEX1.wt ** 0.5)
    HEX1.Lx = np.array([0.7])
    HEX1.Ly = np.array([1.0])
    HEX1.Lz = np.array([1.3])

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
        HEX1.Q = np.reshape(HEX1.eps * HEX1.Cmin * abs(f1.T0[0] - f2.T0[0]), -1)

        t2_air = f2.T0[-1]
        p2_air = f2.p0[-1]
        dp_air = p1_air - p2_air

        t2_oil = f1.T0[-1]
        p2_oil = f1.p0[-1]
        dp_oil = p1_oil - p2_oil

        cost = dp_air + np.abs(HEX1.Q - Q_goal)
        cost = cost[0]
        print(f'sigma: {x[0]}, alpha: {x[1]}, chi: {x[2]}')
        print(
            f'cost: {cost}, dp oil [pa]: {dp_oil[0]}, dp air [bar]: {dp_air[0] * 1e-5}, t2_air: {t2_air[0]},'
            f' t2_oil: {t2_oil[0]}, Q: {HEX1.Q * 1e-3}, procentual pressure loss: {dp_air[0] / p1_air * 100}'
            f' Lx: {0.8}')
    except ValueError:
        cost = 1e12

    return cost


sigma_limits = (0.3, 2.0)
alpha_limits = (0.5, 2.0)
chi_limits = (0.06, 0.3)
#Lx_limits = (0.5, 2.0)

limits = [sigma_limits, alpha_limits, chi_limits]

Q_goal = 1600e3  # heat from piston engine minus heat ejected to fuel

x0 = np.array([1.0, 1.0, 0.07])

# methods that allows bounds (and seem to find global minium): Nelder-Mead, Powell

# NOTE: trust-constr seems to be slow and not find best minimum
# COBYLA doesnt work
# SLSQP doesnt work
# TNC doenst find best minimum
# L-BFGS-B doesnt find best minimum

#sol = minimize(find_optimum, x0, bounds=limits, method='Nelder-Mead')
opt = differential_evolution(find_optimum, bounds=limits)





