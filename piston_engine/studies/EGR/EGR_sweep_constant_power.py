import matplotlib.pyplot as plt
import importlib
import numpy as np
from timeit import default_timer as timer

import sys
sys.path.append("./../../../")



from piston_engine.engine import run_piston_engine  # import the piston engine function
from thermo import fuel_props

# import all the input variables

input_file = "4stroke_EGR"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)


flags = ['sweep'] # to make the function not plot anything


piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'equ_in': d.equ_in,
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
    'mode': d.mode,
}


num = 10
param_name = "equ_in"
params = np.linspace(0.0,0.5, num)

indicated_power = np.zeros(num)
no_concentration = np.zeros(num)
thermal_eff = np.zeros(num)
T_out = np.zeros(num)
intake_massflow = np.zeros(num)
peak_pressure = np.zeros(num)
peak_temperature = np.zeros(num)
far_avg = np.zeros(num) # fuel flow / air flow
heat_loss = np.zeros(num)
exhaust_massflow = np.zeros(num)
hot_zone_peak_temperature = np.zeros(num)
flame_temp = np.zeros(num)
T_sc = np.zeros(num)
p_sc = np.zeros(num)
no_specific = np.zeros(num) # g / kWh
fuel_flow = np.zeros(num)


fuel_type = "jetA"
far_s, LHV = fuel_props(fuel_type)

flags =  ["sweep"]

i = 0

# fuel-air-ratio for no EGR
f_0 = piston_input["far_goal"]

for equ_in in params:
    print(i)


    piston_input["equ_in"] = equ_in
    piston_input["far_goal"] = equ_in * far_s + f_0 * ( 1 + equ_in * far_s)

    start = timer()
    piston_output = run_piston_engine(piston_input, flags)
    end = timer()
    print(f"Time: {end - start}")

    indicated_power[i] = piston_output["indicated power"]
    no_concentration[i] = piston_output["no_ppm"]
    thermal_eff[i] = piston_output["eta_th"]
    T_out[i] = piston_output["T_out"]
    intake_massflow[i] = piston_output["intake massflow"]
    fuel_flow[i] = piston_output["fuel flow"]
    peak_pressure[i] = piston_output["peak pressure"]
    peak_temperature[i] = piston_output["peak temperature"]
    far_avg[i] = piston_output["far"]
    heat_loss[i] = piston_output["heat_loss"]
    exhaust_massflow[i] = piston_output["out_flow"]
    hot_zone_peak_temperature[i] = piston_output["peak temperature hot zone"]
    flame_temp[i] = piston_output["flame temperature"]
    T_sc[i] = piston_output["T start of combustion"]
    p_sc[i] = piston_output["p start of combustion"]
    no_specific[i] = piston_output["nox_spec"] # g/kWh of NO

    i = i + 1



_, ax1 = plt.subplots()

ax1.plot(params, indicated_power*1e-3)
ax1.set_xlabel(f"{param_name}")
ax1.set_ylabel(f"power [kW]")


_, ax2 = plt.subplots()
ax2.plot(params, no_concentration)
ax2.set_xlabel(f"{param_name}")
ax2.set_ylabel(f"NO [ppm]")

_, ax3 = plt.subplots()
ax3.plot(params, thermal_eff)
ax3.set_xlabel(f"{param_name}")
ax3.set_ylabel(f"Thermal efficiency [%]")

_, ax4 = plt.subplots()
ax4.plot(params, T_out)
ax4.set_xlabel(f"{param_name}")
ax4.set_ylabel(f"Exhaust temperature [K]")


_, ax5 = plt.subplots()
ax5.plot(params, intake_massflow)
ax5.set_xlabel(f"{param_name}")
ax5.set_ylabel(f"Intake massflow [kg/s]")

_, ax6 = plt.subplots()
ax6.plot(params, peak_pressure*1e-5)
ax6.set_xlabel(f"{param_name}")
ax6.set_ylabel(f"Peak pressure [bar]")

_, ax7 = plt.subplots()
ax7.plot(params, peak_temperature)
ax7.set_xlabel(f"{param_name}")
ax7.set_ylabel(f"Peak temperature [K]")

_, ax8 = plt.subplots()
ax8.plot(params, far_avg*100)
ax8.set_xlabel(f"{param_name}")
ax8.set_ylabel(f"Average fuel air ratio [%]")

_, ax9 = plt.subplots()
ax9.plot(params, heat_loss*1e-3)
ax9.set_xlabel(f"{param_name}")
ax9.set_ylabel(f"Heat loss [kW]")

_, ax10 = plt.subplots()
ax10.plot(params, exhaust_massflow)
ax10.set_xlabel(f"{param_name}")
ax10.set_ylabel(f"Exhuast mass flow [kg/s]")

_, ax11 = plt.subplots()
ax11.plot(params, hot_zone_peak_temperature)
ax11.set_xlabel(f"{param_name}")
ax11.set_ylabel(f"Hot zone peak temperature [K]")


_, ax12 = plt.subplots()
ax12.plot(params, flame_temp)
ax12.set_xlabel(f"{param_name}")
ax12.set_ylabel(f"Flame temperature [K]")

_, ax13 = plt.subplots()
ax13.plot(params, T_sc)
ax13.set_xlabel(f"{param_name}")
ax13.set_ylabel(f"Temperature at start of combustion [K]")

_, ax14 = plt.subplots()
ax14.plot(params, p_sc*1e-5)
ax14.set_xlabel(f"{param_name}")
ax14.set_ylabel(f"Pressure at start of combustion [bar]")

_, ax15 = plt.subplots()
ax15.plot(params, no_specific)
ax15.set_xlabel(f"{param_name}")
ax15.set_ylabel(f"NO per indicated work [g/kWh]")

_, ax16 = plt.subplots()
ax16.plot(params, fuel_flow*1000)
ax16.set_xlabel(f"{param_name}")
ax16.set_ylabel(f"Fuel flow [g/s]")


plt.show()



