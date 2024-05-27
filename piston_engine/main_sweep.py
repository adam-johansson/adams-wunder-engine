from timeit import default_timer as timer
import matplotlib.pyplot as plt
import importlib

import numpy as np

from engine import run_piston_engine  # import the piston engine function

# import all the input variables

input_file = "4stroke"
# input_file = "nasa_validation"
input_dir = "input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot, output, validation, sweep
flags = ["sweep"]

# chose parameter to investigate
points = 10
#throttles = np.linspace(0.5, 1.0, points)
#pressures = np.linspace(1e5, 15e5, points)
p_ratios = np.linspace(0.7, 3.0, points)
works = []
effs = []
flow = []
i = 1

for p_ratio in p_ratios:
    data = [d.p_in, d.T_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
            d.v_mean, d.lms, d.Twalls, d.ch,
            d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
            d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
            d.cylinders]


    #start = timer()
    T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped = run_piston_engine(data, flags)
    #end = timer()
    # print(f"Runtime of script: {end - start} [s]")
    # print(f"Outlet pressure: {p4 * 1e-5} [bar]")
    # print(f"Outlet temperature: {T4} [K]")
    # print(f"Piston power: {work_piston * 1e-3} [kW]")
    print(f'Data point {i} out of {points}')
    i += 1
    works.append(work_piston * 1e-3)
    effs.append(eff)
    flow.append(air_flow)

fig, ax = plt.subplots()
ax.plot(p_ratios, flow, marker='o')
ax.set_xlabel('Pressure ratio []')
ax.set_ylabel(r'Mass flow of air [kg/s]')
ax.set_title('Mass flow vs pressure ratio')

fig, ax = plt.subplots()
ax.plot(p_ratios, works, marker='o')
ax.set_xlabel('Pressure ratio []')
ax.set_ylabel(r'Shaft power [kW]')
ax.set_title('Engine power vs pressure ratio')

fig, ax = plt.subplots()
ax.plot(p_ratios, effs, marker='o')
ax.set_xlabel('Pressure ratio []')
ax.set_ylabel(r'Thermal efficiency [-]')
ax.set_title('Thermal efficiency vs pressure ratio')
plt.show()
