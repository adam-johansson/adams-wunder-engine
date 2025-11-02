import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.append("./../../../")



from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib

from timeit import default_timer as timer

# Importing input parameters

operating_point = "cruise"

input_file = f"MR_{operating_point}_jetA"
input_dir = "CCE.input.cce_jetA"
path = input_dir + "." + input_file

input_file_pist = "4stroke_jetA"
input_dir_pist = "CCE.input.piston"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)


flags = ["life_hack", "cce"]  # life hack version


constant_F = False


cce_input = {
    "Fn": d.Fn,
    "dTisa": d.dTisa,
    "bpr": d.bpr,
    "T4": d.T4,
    "fpr_outer": d.fpr_outer,
    "Fs_req": d.Fs_req,
    "dp_intake": d.dp_intake,
    "dp_bypass": d.dp_bypass,
    "M": d.M,
    "eta_fan": d.eta_fan,
    "eta_p_hpc": d.eta_p_hpc,
    "eta_p_lpc": d.eta_p_lpc,
    "eta_b": d.eta_b,
    "dPcomb": d.dPcomb,
    "eta_s": d.eta_s,
    "eta_g": d.eta_g,
    "q_ngv": d.q_ngv,
    "bpr_c": d.bpr_c,
    "eta_lpt": d.eta_lpt,
    "cfg_core": d.cfg_core,
    "cfg_bypass": d.cfg_bypass,
    "cd_nozzle": d.cd_nozzle,
    "alt": d.alt,
    "fuel": d.fuel,
    "OPR": d.OPR,
    "PR": d.PR,
    "t_fuel": d.t_fuel,
    "t_tank": d.t_tank,
    "power_offtake": d.power_offtake,
    "surrogate": d.surrogate,
    "second_burner": d.second_burner,
    "pi_pe": d.pi_pe,
    "cr": d.cr,
    "bore": d.bore,
    "far piston": d.far_piston,
    'effectiveness IC': d.eff_IC,
    'dp_inter_compressor': d.dp_inter_compressor,
    "intercooler": d.intercooler,
    "specific": d.specific,
    "v_mean": d.v_mean,
    "start_of_combustion": d.start_of_combustion,
}

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


param_name = "ignition_timing"
params = np.linspace(330,360, num)


SFCs = np.zeros(num)

EI_noxs = np.zeros(num)
EI_noxs_pe = np.zeros(num)
EI_noxs_burner = np.zeros(num)

m_NO_tot = np.zeros(num)

pmaxs = np.zeros(num)
dT_intercoolers = np.zeros(num)
Tmaxs = np.zeros(num)
T_max_twozone = np.zeros(num)

core_effs = np.zeros(num)
thermal_effs = np.zeros(num)
transmission_effs = np.zeros(num)
propulsive_effs = np.zeros(num)
overall_effs = np.zeros(num)

specific_thrusts = np.zeros(num)
specific_powers = np.zeros(num)

core_powers = np.zeros(num)
piston_powers = np.zeros(num)

hot_bypass_thrusts = np.zeros(num)
cold_bypass_thrusts = np.zeros(num)
core_thrusts = np.zeros(num)

piston_fuelflow = np.zeros(num)
burner_fuelflow = np.zeros(num)

cool_ngv = np.zeros(num)
cool_rotor = np.zeros(num)
m_core = np.zeros(num)

bprs = np.zeros(num)
bores = np.zeros(num)

piston_bprs = np.zeros(num)
piston_power_spec = np.zeros(num)

meta_model = "placeholder"


start = timer()
i = 0
for sc in params:

    lap1 = timer()

    cce_input["start_of_combustion"] = sc
    print(sc)

    # run once to get specific power and linear output temperature from piston engine
    cce_input["life_hack"] = "Simulate"
    # bpr 20 will almost certainly work
    cce_input["bpr"] = 20
    output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

    # input simulation data for bpr matching
    piston_input["k_m"] = output_dict["k_m"]
    piston_input["k0_T"] = output_dict["k0_T"]
    piston_input["k1_T"] = output_dict["k1_T"]
    piston_input["k0_H"] = output_dict["k0_H"]
    piston_input["k1_H"] = output_dict["k1_H"]
    piston_input["piston_specific_power"] = output_dict["piston_specific_power"]

    # no simulation just quick evaluations to find BPR to match thrust
    cce_input["life_hack"] = "Express"

    dict = auxiliaries.run_cce_bpr(cce_input, piston_input, meta_model)

    if dict["error"]:
        # if no BPR is found that matches thrust
        bprs[i] = dict["bpr"]


    else:

        cce_input["bpr"] = dict["bpr"][0]
        cce_input["bore"] = dict["bore_match"]

        bprs[i] = dict["bpr"][0]


        # final simulation with known bore and BPR to get all info and especially NOX
        cce_input["life_hack"] = "Simulate_final"
        #print("Final simulation")
        output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

        #print(f"Mass flow: {output_dict["mass flow"]} kg/s")
        #print(f"Specific thrust: {output_dict["specific thrust"]} N/kg/s")
        #print(f"Thrust: {output_dict["thrust"] * 1e-3} kN")


        SFCs[i] = output_dict["sfc"]
        pmaxs[i] = output_dict["p_max"]
        EI_noxs[i] = output_dict["EI_nox"]
        EI_noxs_pe[i] = output_dict["EI_nox_PE"]
        EI_noxs_burner[i] = output_dict["EI_nox_burner"]

        m_NO_tot[i] = output_dict["m_NO_tot"]

        core_effs[i] = output_dict["core efficiency"]
        transmission_effs[i] = output_dict["transmission efficiency"]
        thermal_effs[i] = output_dict["thermal efficiency"]
        propulsive_effs[i] = output_dict["propulsive efficiency"]
        overall_effs[i] = output_dict["overall efficiency"]

        specific_thrusts[i] = output_dict["specific thrust"]
        specific_powers[i] = output_dict["core specific power"]
        core_powers[i] = output_dict["core power"]
        dT_intercoolers[i] = output_dict["dT intercooler"]
        Tmaxs[i] = output_dict["T_max"]
        T_max_twozone[i] = output_dict["T_max_twozone"]

        hot_bypass_thrusts[i] = output_dict["hot bypass thrust"]
        cold_bypass_thrusts[i] = output_dict["cold bypass thrust"]
        core_thrusts[i] = output_dict["core thrust"]

        piston_fuelflow[i] = output_dict["piston fuelflow"]
        burner_fuelflow[i] = output_dict["burner fuelflow"]

        cool_ngv[i] = output_dict["m_cool_ngv"]
        cool_rotor[i] = output_dict["m_cool_rotor"]
        m_core[i] = output_dict["core mass flow"]

        bores[i] = output_dict["bore"]
        piston_bprs[i] = output_dict["bpr_piston"]
        piston_power_spec[i] = output_dict["piston_specific_power"]
        piston_powers[i] = output_dict["piston_power"]

    lap2 = timer()
    print(f"Simulation time for 1 point: {lap2 - lap1} seconds")

    i = i+1

end = timer()
print(f"Total simulation time for {i} evaluation points: {end - start} seconds")


#little bit of post processing

# engine displacement
disp = 24 * bores*bores*bores*np.pi*1000/4


# core power per liter of piston engine
# disp is already in liter
core_powers_per_displacement = (core_powers / disp) * 1e-3

# piston power per liter
piston_powers_per_liter = (piston_powers / disp) * 1e-3

# g NO per kWh of core energy
# m_NO_tot is kg/s
# core_power is in W

# we want g/kWh

NO_per_power = (1000 * m_NO_tot * 3600) / (core_powers * 1e-3)


_, ax1 = plt.subplots()

ax1.plot(params, SFCs*1e6)
ax1.set_xlabel(f"{param_name}")
ax1.set_ylabel(f"SFC [mg/Ns]")


_, ax2 = plt.subplots()

ax2.plot(params, transmission_effs*100,label="Transmission", linewidth="2")
ax2.plot(params, propulsive_effs*100,label="Propulsive", linewidth="2")
ax2.plot(params, core_effs*100,label="Core", linewidth="2")
ax2.plot(params, thermal_effs*100,label="Thermal", linewidth="2")
ax2.plot(params, overall_effs*100,label="Overall", linewidth="2")
ax2.set_xlabel(f"{param_name}")
ax2.legend()

_, ax3 = plt.subplots()
ax3.plot(params, specific_thrusts)
ax3.set_xlabel(f"{param_name}")

_, ax4 = plt.subplots()
ax4.plot(params, core_effs*100,label="Core", linewidth="2")
ax4.plot(params, thermal_effs*100,label="Thermal", linewidth="2")
ax4.set_xlabel(f"{param_name}")
ax4.legend()

_, ax5 = plt.subplots()
ax5.plot(params, transmission_effs*100,label="Transmission", linewidth="2")
ax5.plot(params, propulsive_effs*100,label="Propulsive", linewidth="2")
ax5.set_xlabel(f"{param_name}")
ax5.legend()

_, ax6 = plt.subplots()
ax6.plot(params, hot_bypass_thrusts*1e-3,label="Hot", linewidth="2")
ax6.plot(params, cold_bypass_thrusts*1e-3,label="Cold", linewidth="2")
ax6.plot(params, core_thrusts*1e-3,label="Core", linewidth="2")
ax6.set_xlabel(f"{param_name}")
ax6.legend()

_, ax7 = plt.subplots()
ax7.plot(params, piston_fuelflow,label="Piston", linewidth="2")
ax7.plot(params, burner_fuelflow,label="Burner", linewidth="2")
ax7.plot(params, burner_fuelflow + piston_fuelflow,label="Total", linewidth="2")
ax7.set_xlabel(f"{param_name}")
ax7.legend()

_, ax8 = plt.subplots()
ax8.plot(params, 100*piston_fuelflow / (piston_fuelflow + burner_fuelflow),label="Piston", linewidth="2")
ax8.plot(params, 100*burner_fuelflow/ (piston_fuelflow + burner_fuelflow),label="Burner", linewidth="2")
ax8.set_xlabel(f"{param_name}")
ax8.legend()


_, ax9 = plt.subplots()
ax9.plot(params, bprs)
ax9.set_xlabel(f"{param_name}")
ax9.set_ylabel("BPR")

_, ax10 = plt.subplots()
ax10.plot(params, pmaxs*1e-5)
ax10.set_xlabel(f"{param_name}")
ax10.set_ylabel("max pressure")

_, ax11 = plt.subplots()
ax11.plot(params, EI_noxs, label="Total")
ax11.plot(params, EI_noxs_pe, label="PE")
ax11.plot(params, EI_noxs_burner, label="Burner")
ax11.set_xlabel(f"{param_name}")
ax11.set_ylabel("EI NOx [g/kg]")
ax11.legend()

_, ax12 = plt.subplots()
ax12.plot(params, thermal_effs*100)
ax12.set_xlabel(f"{param_name}")
ax12.set_ylabel("Thermal efficiency [percent]")

_, ax13 = plt.subplots()
ax13.plot(params, specific_powers*1e-3)
ax13.set_xlabel(f"{param_name}")
ax13.set_ylabel("Specific power core [kW/kg/s]")

_, ax14 = plt.subplots()
ax14.plot(dT_intercoolers, specific_powers*1e-3)
ax14.set_xlabel(f"dT intercooler")
ax14.set_ylabel("Specific power core [kW/kg/s]")

_, ax15 = plt.subplots()
ax15.plot(dT_intercoolers, EI_noxs)
ax15.set_xlabel(f"dT intercooler")
ax15.set_ylabel("EI NOx [g/kg]")

_, ax16 = plt.subplots()
ax16.plot(dT_intercoolers, thermal_effs*100)
ax16.set_xlabel(f"dT intercooler")
ax16.set_ylabel("Thermal efficiency [percent]")


_, ax17 = plt.subplots()
ax17.plot(params, dT_intercoolers)
ax17.set_xlabel(f"{param_name}")
ax17.set_ylabel("dT intercooler [K]")

_, ax18 = plt.subplots()
ax18.plot(params, Tmaxs, label="1zone")
ax18.plot(params, T_max_twozone, label="2zone")
ax18.set_xlabel(f"{param_name}")
ax18.set_ylabel("T max [K]")
ax18.legend()

_, ax19 = plt.subplots()
ax19.plot(params, cool_ngv, label="NGV")
ax19.plot(params, cool_rotor, label="Rotor")
ax19.plot(params, cool_ngv + cool_rotor, label="Total")
ax19.set_xlabel(f"{param_name}")
ax19.set_ylabel("m cool [kg/s]")
ax19.legend()

_, ax20 = plt.subplots()
ax20.plot(params, (cool_ngv + cool_rotor)/m_core )
ax20.set_xlabel(f"{param_name}")
ax20.set_ylabel("fraction cool [-]")


_, ax21 = plt.subplots()
ax21.plot(params, bores*1000 )
ax21.set_xlabel(f"{param_name}")
ax21.set_ylabel("bore [mm]")


_, ax22 = plt.subplots()
ax22.plot(params, disp )
ax22.set_xlabel(f"{param_name}")
ax22.set_ylabel("displacement [liter]")

_, ax23 = plt.subplots()
ax23.plot(params, piston_bprs)
ax23.set_xlabel(f"{param_name}")
ax23.set_ylabel("bpr piston [-]")

_, ax24 = plt.subplots()
ax24.plot(params, piston_power_spec*1e-3)
ax24.set_xlabel(f"{param_name}")
ax24.set_ylabel("piston power per massflow [kW/kg/s]")

_, ax25 = plt.subplots()
ax25.plot(params, core_powers_per_displacement)
ax25.set_xlabel(f"{param_name}")
ax25.set_ylabel("core power per liter [kW/liter]")

_, ax26 = plt.subplots()
ax26.plot(params, piston_powers_per_liter)
ax26.set_xlabel(f"{param_name}")
ax26.set_ylabel("piston power per liter [kW/liter]")

_, ax27 = plt.subplots()
ax27.plot(params, NO_per_power)
ax27.set_xlabel(f"{param_name}")
ax27.set_ylabel("NO flow per power [g/kWh]")

plt.show()

EI_NO = np.vstack((params, EI_noxs)).transpose()
# g/kWh
NO_per_power = np.vstack((params, NO_per_power)).transpose()
eff = np.vstack((params, thermal_effs*100)).transpose()
power = np.vstack((params, specific_powers*1e-3)).transpose()
peak_pressure = np.vstack((params, pmaxs*1e-5)).transpose()
fuel_consumption = np.vstack((params, SFCs*1e6)).transpose()
bypass_ratios = np.vstack((params, bprs)).transpose()
bore_diameters = np.vstack((params, bores*1000)).transpose()
displacements = np.vstack((params, disp)).transpose()
piston_bypass_ratios = np.vstack((params, piston_bprs)).transpose()
piston_specific_power = np.vstack((params, piston_power_spec*1e3)).transpose()
peak_temp = np.vstack((params, Tmaxs)).transpose()
peak_temp_twozone = np.vstack((params, T_max_twozone)).transpose()

# kW per liter
core_power_per_displacement = np.vstack((params, core_powers_per_displacement)).transpose()
piston_power_per_displacement = np.vstack((params, piston_powers_per_liter)).transpose()

# save output for plotting in latex
np.savetxt(f"./results/{operating_point}/{param_name}/EI_NOX.dat", EI_NO, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/NOX_per_power.dat", NO_per_power, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/efficiency.dat", eff, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/spec_power.dat", power, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/pmax.dat", peak_pressure, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/Tmax.dat", peak_temp, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/Tmax_twozone.dat", peak_temp_twozone, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/SFC.dat", fuel_consumption, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/BPR.dat", bypass_ratios, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/bore.dat", bore_diameters, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/displacement.dat", displacements, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/piston_bpr.dat", piston_bypass_ratios, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/piston_spec_power.dat", piston_specific_power, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/core_power_per_liter.dat", core_power_per_displacement, fmt="%.5f")
np.savetxt(f"./results/{operating_point}/{param_name}/piston_power_per_liter.dat", piston_power_per_displacement, fmt="%.5f")

