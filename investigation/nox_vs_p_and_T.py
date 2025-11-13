import numpy as np
import importlib 
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import sys
sys.path.append("./../")

from thermo import fuel_props, JETA_L, H2, mixture

from piston_engine.engine import run_piston_engine

flags = ["sweep"]

# Importing input parameters
input_file_pist = "4stroke_standard"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d_p = importlib.import_module(path_pist)

piston_input = {
    'p_in': d_p.p_in,
    'T_in': d_p.T_in,
    'p_ratio': d_p.p_ratio,
    'cycle': d_p.cycle,
    'cooling': d_p.cooling,
    'opposed': d_p.opposed,
    'cr': d_p.cr,
    'bore': d_p.d,
    'bsr': d_p.bsr,
    'v_mean': d_p.v_mean,
    'lms': d_p.lms,
    'Twalls': d_p.Twalls,
    'ch': d_p.ch,
    'valve_timings': d_p.valve_timings,
    'n_valve': d_p.n_valve,
    'lv_max': d_p.lv_max,
    'cd': d_p.cd,
    'eta_c': d_p.eta_c,
    'mf_tot': d_p.mf_tot,
    'wa': d_p.wa,
    'wm': d_p.wm,
    'm_wiebe': d_p.m_wiebe,
    'phi_sc': d_p.phi_sc,
    'phi_cd': d_p.phi_cd,
    'T_fuel': d_p.T_fuel,
    'p_fuel': d_p.p_fuel,
    'it': d_p.it,
    'wiebe_type': d_p.wiebe_type,
    'valve_type': d_p.valve_type,
    'far_goal': d_p.far_goal,
    'cylinders': d_p.cylinders,
    'fuel': d_p.fuel,
    'c1': d_p.c1,
    'c4': d_p.c4,
    'c5': d_p.c5,
    'premixed': d_p.premixed,
}

num = 10
far_stoich, _ = fuel_props("jetA")
p_in_array = np.linspace(1e5,5e5, num)



nox_array = np.zeros(num)
T_max_two_zone_array = np.zeros(num)
T_max_single_zone_array = np.zeros(num)
T_flame_array = np.zeros(num)
T_sc_array = np.zeros(num)
p_max_array = np.zeros(num)
p_sc_array = np.zeros(num)



p0 = 1e5
T0 = 300

i = 0

start = timer()

for p_in in p_in_array: 

    lap1 = timer()

    print(i)

    T_in = T0*(p_in/p0)**(0.4/1.4)

    piston_input["p_in"] = p_in
    piston_input["T_in"] = T_in

    piston_output = run_piston_engine(piston_input, flags)

    lap2 = timer()
    print(f"Simulation time for 1 point: {lap2 - lap1} seconds")



    nox_array[i] = piston_output["no_ppm"]
    T_max_two_zone_array[i] = piston_output["peak temperature hot zone"]
    T_max_single_zone_array[i] = piston_output["peak temperature"]
    p_max_array[i] = piston_output["peak pressure"]
    T_flame_array[i] = piston_output["flame temperature"]
    T_sc_array[i] = piston_output["T start of combustion"]
    p_sc_array[i] = piston_output["p start of combustion"]

    i = i + 1

end = timer()
print(f"Total simulation time for {i} evaluation points: {end - start} seconds")

_, ax1 = plt.subplots()

ax1.plot(p_in_array*1e-5, nox_array)
ax1.set_xlabel("Inlet pressure [bar]")
ax1.set_ylabel("NO [ppm]")

_, ax2 = plt.subplots()

ax2.plot(p_in_array*1e-5, T_max_two_zone_array)
ax2.set_xlabel("Inlet pressure [bar]")
ax2.set_ylabel("T_max hot zone [K]")


_, ax3 = plt.subplots()
ax3.plot(p_in_array*1e-5, T_max_single_zone_array)
ax3.set_xlabel("Inlet pressure [bar]")
ax3.set_ylabel("T_max single zone [K]")

_, ax4 = plt.subplots()
ax4.plot(p_in_array*1e-5, p_max_array*1e-5)
ax4.set_xlabel("Inlet pressure [bar]")
ax4.set_ylabel("peak pressure [bar]")


_, ax5 = plt.subplots()
ax5.plot(p_in_array*1e-5, T_flame_array)
ax5.set_xlabel("Inlet pressure [bar]")
ax5.set_ylabel("flame temperature [K]")

_, ax6 = plt.subplots()
ax6.plot(p_in_array*1e-5, T_sc_array)
ax6.set_xlabel("Inlet pressure [bar]")
ax6.set_ylabel("T start of combustion [K]")

_, ax7 = plt.subplots()
ax7.plot(p_in_array*1e-5, p_sc_array*1e-5)
ax7.set_xlabel("Inlet pressure [bar]")
ax7.set_ylabel("p start of combustion [bar]")



plt.show()





