import numpy as np

from piston_engine.src.piston.nox_model import nox_calculations

num = 100
T_z1 = np.linspace(1500,1700, num)
m_z1 = np.linspace(0.09,0.1, num)
p_z1 = np.linspace(10e5,11e5, num)
V_z1 = np.linspace(0.004, 0.005, num)

fuel_type = "jetA"

# air-fuel ratio in burned zone
lambda_z1 = 1

# crank angle
phi_z1 = np.linspace(0, 2*np.pi, num)

# rpm
rpm = 3000

# total mass outflow during one cycle
m_out_total = 0.5

# total average fuel-air-equivalence ratio
equ = 0.5

nox = nox_calculations(T_z1, m_z1, p_z1, V_z1, fuel_type, lambda_z1, phi_z1, rpm,
                                 m_out_total, equ)