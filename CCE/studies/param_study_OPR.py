import matplotlib.pyplot as plt
import numpy as np

from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib
from neural_network.src import load_ANN

from timeit import default_timer as timer

# Importing input parameters

input_file = "MR_TOC_jetA"
input_dir = "CCE.input.cce_jetA"
path = input_dir + "." + input_file

input_file_pist = "4stroke_jetA"
input_dir_pist = "CCE.input.piston"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)

#flags = ["single", "print_output", "conventional"]  # normal case
#flags = ["single", "print_output", "cce"]  # normal case
flags = ['single', "cce"] # for matching thrust/no plots
#flags = ['sweep']
#flags = ['optim', "cce"]

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


# Load the trained model
meta_model = load_ANN("../meta_models/jetA_128_2_pinn.pth")
meta_model.double()
print(meta_model)

num = 100

param_name = "OPR"
params = np.linspace(10,20, num)

SFCs = np.zeros(num)
EI_noxs = np.zeros(num)

pmaxs = np.zeros(num)
dT_intercoolers = np.zeros(num)
Tmaxs = np.zeros(num)

core_effs = np.zeros(num)
thermal_effs = np.zeros(num)
transmission_effs = np.zeros(num)
propulsive_effs = np.zeros(num)
overall_effs = np.zeros(num)

specific_thrusts = np.zeros(num)
specific_powers = np.zeros(num)


hot_bypass_thrusts = np.zeros(num)
cold_bypass_thrusts = np.zeros(num)
core_thrusts = np.zeros(num)

piston_fuelflow = np.zeros(num)
burner_fuelflow = np.zeros(num)

bprs = np.zeros(num)


i = 0
for OPR in params:

    cce_input["OPR"] = OPR
    print(OPR)


    dict = auxiliaries.run_cce_bpr(cce_input, piston_input, meta_model)

    if dict["error"]:
        bprs[i] = dict["bpr"]

    else:

        bprs[i] = dict["bpr"]

        output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

        #print(f"Mass flow: {output_dict["mass flow"]} kg/s")
        #print(f"Specific thrust: {output_dict["specific thrust"]} N/kg/s")
        #print(f"Thrust: {output_dict["thrust"] * 1e-3} kN")


        SFCs[i] = output_dict["sfc"]
        pmaxs[i] = output_dict["p_max"]
        EI_noxs[i] = output_dict["EI_nox"]

        core_effs[i] = output_dict["core efficiency"]
        transmission_effs[i] = output_dict["transmission efficiency"]
        thermal_effs[i] = output_dict["thermal efficiency"]
        propulsive_effs[i] = output_dict["propulsive efficiency"]
        overall_effs[i] = output_dict["overall efficiency"]

        specific_thrusts[i] = output_dict["specific thrust"]
        specific_powers[i] = output_dict["core specific power"]
        dT_intercoolers[i] = output_dict["dT intercooler"]
        Tmaxs[i] = output_dict["T_max"]

        hot_bypass_thrusts[i] = output_dict["hot bypass thrust"]
        cold_bypass_thrusts[i] = output_dict["cold bypass thrust"]
        core_thrusts[i] = output_dict["core thrust"]

        piston_fuelflow[i] = output_dict["piston fuelflow"]
        burner_fuelflow[i] = output_dict["burner fuelflow"]


    i = i+1





_, ax1 = plt.subplots()

ax1.plot(params, SFCs*1e6)
ax1.set_xlabel(f"{param_name}")


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
ax11.plot(params, EI_noxs)
ax11.set_xlabel(f"{param_name}")
ax11.set_ylabel("EI NOx [g/kg]")

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
ax18.plot(params, Tmaxs)
ax18.set_xlabel(f"{param_name}")
ax18.set_ylabel("T max [K]")


plt.show()

NO = np.vstack((params, EI_noxs)).transpose()
eff = np.vstack((params, thermal_effs*100)).transpose()
power = np.vstack((params, specific_powers*1e-3)).transpose()

# save output for plotting in latex
np.savetxt(f"./results/NOX_{param_name}.dat", NO, fmt="%.5f")
np.savetxt(f"./results/efficiency_{param_name}.dat", eff, fmt="%.5f")
np.savetxt(f"./results/power_{param_name}.dat", power, fmt="%.5f")

