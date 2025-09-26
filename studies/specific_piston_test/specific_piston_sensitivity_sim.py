import matplotlib.pyplot as plt
from thermo import fuel_props
from piston_engine.engine import run_piston_engine


import numpy as np
from neural_network.src import load_ANN

# input
pin = 10e5
Tin = 500
cr = 10
p_ratio = 1.0
v_mean = 14
T_fuel = 400

far34 = 0.04

fuel_type = "jetA"


num = 10
bores = np.linspace(0.1,0.2,num)

specific_powers = np.zeros(num)
m_ins = np.zeros(num)
heat_losses = np.zeros(num)
p_maxs = np.zeros(num)
T_maxs = np.zeros(num)
T_outs = np.zeros(num)
noxs = np.zeros(num)

displacements = np.zeros(num)

i = 0

for bore in bores:

    piston_input = np.atleast_2d(
    np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far34])
    )

    # use meta model to get outputs from the piston engine
    output = meta_model.inference(piston_input)[0]

    # mass flow of air into the engine
    m_in = output[7]

    # specific power and specific heat loss
    indicated_power = output[0] / m_in
    heat_loss = output[1] / m_in
    nox_ppm = output[2]
    p_tdc = output[3]
    p_max = output[4]
    T_max = output[5]
    T34 = output[6]
    p34 = pin * p_ratio

    far_s, LHV = fuel_props(fuel_type)

    #specific fuel flow (since fuel_flow_tot = far34 * m_in)
    fuel_flow = far34 * m_in

    specific_powers[i] = indicated_power
    m_ins[i] = m_in
    heat_losses[i] = heat_loss
    p_maxs[i] = p_max
    T_maxs[i] = T_max
    T_outs[i] = T34
    noxs[i] = nox_ppm

    displacements[i] = bore * np.pi * (bore/2)**2
    i = i + 1




param_name = "bore"

_, ax1 = plt.subplots()
ax1.plot(bores, specific_powers*1e-3)
ax1.set_xlabel(f"{param_name}")
ax1.set_ylabel(f"Specific power [kW/kg/s]")


_, ax2 = plt.subplots()
ax2.plot(bores, m_ins)
ax2.set_xlabel(f"{param_name}")
ax2.set_ylabel(f"massflow in [kg/s]")



_, ax3 = plt.subplots()
ax3.plot(bores, heat_losses*1e-3)
ax3.set_xlabel(f"{param_name}")
ax3.set_ylabel(f"Specific heat loss [kW/kg/s]")

_, ax4 = plt.subplots()
ax4.plot(displacements*1000, m_ins)
ax4.set_xlabel(f"displacement [l]")
ax4.set_ylabel(f"massflow in [kg/s]")

_, ax5 = plt.subplots()
ax5.plot(displacements*1000, (displacements/m_ins)*1000/m_ins)
ax5.set_xlabel(f"displacement [l]")
ax5.set_ylabel(f"spec displacement [l/kg/s]")

_, ax6 = plt.subplots()
ax6.plot(m_ins,displacements*1000, )
ax6.set_ylabel(f"displacement [l]")
ax6.set_xlabel(f"massflow in [kg/s]")

_, ax7 = plt.subplots()
ax7.plot(m_ins,bores*bores)
ax7.set_ylabel(f"bore**2")
ax7.set_xlabel(f"massflow in [kg/s]")
ax7.grid()


plt.show()



