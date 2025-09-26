import matplotlib.pyplot as plt
from thermo import fuel_props
from piston_engine.engine import run_piston_engine
import importlib


import numpy as np
from neural_network.src import load_ANN

### LOAD THE MODELS

input_file_pist = "4stroke_sampling"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist
d = importlib.import_module(path_pist)
piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'p_ratio': d.p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': d.cr,
    'bore': d.d,
    'bsr': d.bsr,
    'v_mean': d.v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': d.lv_max,
    'cd': d.cd,
    'eta_c': d.eta_c,
    'mf_tot': d.mf_tot,
    'wa': d.wa,
    'wm': d.wm,
    'm_wiebe': d.m_wiebe,
    'phi_sc': d.phi_sc,
    'phi_cd': d.phi_cd,
    'T_fuel': d.T_fuel,
    'p_fuel': d.p_fuel,
    'it': d.it,
    'wiebe_type': d.wiebe_type,
    'valve_type': d.valve_type,
    'far_goal': d.far_goal,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'premixed': d.premixed,
}

# input
pin = 10e5
Tin = 500
cr = 10
p_ratio = 1.0
v_mean = 14
T_fuel = 400

far34 = 0.04

fuel_type = "jetA"


num = 30
bores = np.linspace(0.05,0.3,num)

specific_powers = np.zeros(num)
m_ins = np.zeros(num)
heat_losses = np.zeros(num)
p_maxs = np.zeros(num)
T_maxs = np.zeros(num)
T_outs = np.zeros(num)
noxs = np.zeros(num)

displacements = np.zeros(num)

flags = ["sweep"]

i = 0

for bore in bores:
    print(i)
    piston_input["bore"] = bore  # fuel air ratio
    (
        T34,
        _,
        _,
        m_in,
        p_max,
        T_max,
        _,
        _,
        indicated_power,
        _,
        _,
        heat_loss,
        p_tdc,
        _,
        nox_ppm,
        _,
        EI_nox,
        _,
        nox_spec,
    ) = run_piston_engine(piston_input, flags)


    far_s, LHV = fuel_props(fuel_type)

    #specific fuel flow (since fuel_flow_tot = far34 * m_in)
    fuel_flow = far34 * m_in

    specific_powers[i] = indicated_power / m_in
    m_ins[i] = m_in
    heat_losses[i] = heat_loss / m_in
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



