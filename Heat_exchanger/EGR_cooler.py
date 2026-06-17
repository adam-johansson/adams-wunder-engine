import sys
sys.path.append("./../")



from GenHex_Simple import GenHex_single
import numpy as np
from CoolProp.CoolProp import PropsSI
from thermo import mixture


# Script for running
f1 = type("f1", (), {})()
f2 = type("f2", (), {})()
f3 = type("f3", (), {})()
HEX1 = type("HEX1", (), {})()
HEX2 = type("HEX2", (), {})()


# ----------------------------------Fluid inputs----------------------------------

t1_EGR = 1200
p1_EGR = 12e5
m_EGR = 4
equ_EGR = 0.3
h1_EGR, _, _, _, _, _, _, _ = mixture(t=t1_EGR, p=p1_EGR, equivalence_ratio=equ_EGR, fuel_type="jetA")


t1_air = 700
p1_air = 10e5
m_air = 8
h1_air, _, _, _, _, _, _, _ = mixture(t=t1_air, p=p1_air)


# EGR gas
f1.T01 = np.array([t1_EGR])
f1.p01 = np.array([p1_EGR])
f1.mdot = np.array([m_EGR])
f1.fluid = "INCOMP::PNF2"  # some kind of heat transfer mineral oil
f1.dataSource = "Coolprop_incomp"

# Pure air
f2.T01 = np.array([t1_h2])
f2.p01 = np.array([p1_h2])
f2.mdot = np.array([m_h2])
f2.fluid = "Hydrogen"
f2.dataSource = "coolprop"

# Geometrical
HEX1.wt = 0.0005
HEX1.FinAR = 0.01 / (HEX1.wt**0.5)
HEX1.Lx = np.array([0.1])
HEX1.Ly = np.array([0.1])
HEX1.Lz = np.array([0.1])

HEX1.correlation = "KL"

HEX1.f1_flowdir = "Ly"
HEX1.f1_L = 0.01  # dont change
HEX1.f2_flowdir = "Lx"
HEX1.f2_L = 0.01  # dont change

# Material
HEX1.density = 2840  # Alu 2219
HEX1.k = 120  # Conductivity # Alu 2219

# ----------------------------------Generalized geometry input ----------------------------------
HEX1.sigma_r = np.array([1.0])
HEX1.alpha_r = np.array([1.0])
HEX1.chi = np.array([0.1])

f1, f2, HEX1 = GenHex_single(f1, f2, HEX1, verbose=False)

print("fluid 1 temperatures", f1.T0, "fluid 1 pressures", f1.p0)
print("HEX effectiveness", HEX1.eps)

HEX1.Q = np.reshape(HEX1.eps * HEX1.Cmin * abs(f1.T0[0] - f2.T0[0]), -1)

t2_oil = f1.T0[-1]
p2_oil = f1.p0[-1]

t2_h2 = f2.T0[-1]
p2_h2 = f2.p0[-1]


h2_oil = PropsSI("Hmass", "T", t2_oil, "P", p2_oil, "INCOMP::PNF2")

h2_h2 = PropsSI("Hmass", "T", t2_h2, "P", p2_h2, "Hydrogen")


Q1 = m_oil * (h1_oil - h2_oil)
Q2 = m_h2 * (h1_h2 - h2_h2)

dT1 = t1_oil - t2_oil
dp1 = p1_oil - p2_oil

dT2 = t1_h2 - t2_h2
dp2 = p1_h2 - p2_h2

print(f"Heating power: {HEX1.Q * 1e-3} kW")
print(f"Heating power oil: {Q1 * 1e-3} kW")
print(f"Heating power h2: {Q2 * 1e-3} kW")

print(f"T1 oil: {f1.T0[0]}")
print(f"dT oil: {dT1}")
print(f"T2 oil: {f1.T0[-1]}")
print(f"dp oil: {dp1}")
print(f"T1 hydrogen: {f2.T0[0]}")
print(f"dT hydrogen: {dT2}")
print(f"T2 hydrogen: {f2.T0[-1]}")
print(f"dp hydrogen: {dp2}")
