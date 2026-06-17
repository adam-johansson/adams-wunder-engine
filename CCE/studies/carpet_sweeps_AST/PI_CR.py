import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.append("./../../../")




from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib

from timeit import default_timer as timer

# Importing input parameters


operating_point = "TOC"

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
    "ratio IC": d.ratio_IC,
    "piston_mode": d.piston_mode,
    "LPT_eff_type": d.LPT_eff_type,
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



param_name = "PI_CR"
params_1 = np.arange(0.9,1.41,0.1)
params_2 = np.arange(5,9.1,1)


num1 = np.size(params_1)
num2 = np.size(params_2)

SFCs = np.zeros((num1,num2))

EI_noxs = np.zeros((num1,num2))
m_noxs_pe = np.zeros((num1,num2))
m_noxs_burner = np.zeros((num1,num2))

m_NO_tot = np.zeros((num1,num2))
specific_nox = np.zeros((num1,num2))

pmaxs = np.zeros((num1,num2))
dT_intercoolers = np.zeros((num1,num2))
Tmaxs = np.zeros((num1,num2))
T_max_twozone = np.zeros((num1,num2))
T4s = np.zeros((num1,num2))
T34s = np.zeros((num1,num2))
T35s = np.zeros((num1,num2))

core_effs = np.zeros((num1,num2))
thermal_effs = np.zeros((num1,num2))
transmission_effs = np.zeros((num1,num2))
propulsive_effs = np.zeros((num1,num2))
overall_effs = np.zeros((num1,num2))
gg_effs = np.zeros((num1,num2))

specific_thrusts = np.zeros((num1,num2))
specific_powers = np.zeros((num1,num2))

core_powers = np.zeros((num1,num2))
gg_powers = np.zeros((num1,num2))
gg_mass_spec_powers = np.zeros((num1,num2))
gg_disp_spec_powers = np.zeros((num1,num2))
cooling_ratios = np.zeros((num1,num2))

hot_bypass_thrusts = np.zeros((num1,num2))
cold_bypass_thrusts = np.zeros((num1,num2))
core_thrusts = np.zeros((num1,num2))

piston_fuelflow = np.zeros((num1,num2))
burner_fuelflow = np.zeros((num1,num2))

cool_ngv = np.zeros((num1,num2))
cool_rotor = np.zeros((num1,num2))
m_core = np.zeros((num1,num2))

bprs = np.zeros((num1,num2))
bores = np.zeros((num1,num2))

piston_bprs = np.zeros((num1,num2))
piston_power_spec = np.zeros((num1,num2))


piston_powers = np.zeros((num1,num2))
piston_heatloss = np.zeros((num1,num2))
piston_powers_indicated = np.zeros((num1,num2))
heatloss_percentage = np.zeros((num1,num2))
friction_percentage = np.zeros((num1,num2))


meta_model = "placeholder"

# save orignal efficiency values
eta_p_hpc_0 = cce_input["eta_p_hpc"] 
eta_lpt_0 = cce_input["eta_lpt"]

start = timer()
i = 0
for pi in params_1:
    j = 0
    for cr in params_2:


        lap1 = timer()

        cce_input["cr"] = cr
        cce_input["pi_pe"] = pi
        print(pi, cr)

        cce_input["eta_p_hpc"] = eta_p_hpc_0
        cce_input["eta_lpt"] = eta_lpt_0

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
        cce_input["eta_p_hpc"] = output_dict["eta_hpc"]
        cce_input["eta_lpt"] = output_dict["eta_lpt"]

        # no simulation just quick evaluations to find BPR to match thrust
        cce_input["life_hack"] = "Express"

        dict = auxiliaries.run_cce_bpr(cce_input, piston_input, meta_model)

        if dict["error"]:
            # if no BPR is found that matches thrust
            bprs[i,j] = dict["bpr"]


        else:

            cce_input["bpr"] = dict["bpr"][0]
            cce_input["bore"] = dict["bore_match"]

            bprs[i,j] = dict["bpr"][0]


            # final simulation with known bore and BPR to get all info and especially NOX
            cce_input["life_hack"] = "Simulate_final"
            #print("Final simulation")
            output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

            #print(f"Mass flow: {output_dict["mass flow"]} kg/s")
            #print(f"Specific thrust: {output_dict["specific thrust"]} N/kg/s")
            #print(f"Thrust: {output_dict["thrust"] * 1e-3} kN")


            SFCs[i,j] = output_dict["sfc"]
            pmaxs[i,j] = output_dict["p_max"]
            EI_noxs[i,j] = output_dict["EI_nox"]
            m_noxs_pe[i,j] = output_dict["m_nox_PE"]
            m_noxs_burner[i,j] = output_dict["m_nox_burner"]

            m_NO_tot[i,j] = output_dict["m_NO_tot"]
            specific_nox[i,j] = output_dict["thrust specific nox"]

            core_effs[i,j] = output_dict["core efficiency"]
            transmission_effs[i,j] = output_dict["transmission efficiency"]
            thermal_effs[i,j] = output_dict["thermal efficiency"]
            propulsive_effs[i,j] = output_dict["propulsive efficiency"]
            overall_effs[i,j] = output_dict["overall efficiency"]
            
            
            gg_effs[i,j] = output_dict["gg efficiency"]
            gg_powers[i,j] = output_dict["gg_power"]
            gg_mass_spec_powers[i,j] = output_dict["gg_mass_specific_power"]
            gg_disp_spec_powers[i,j] = output_dict["gg_disp_specific_power"]
            cooling_ratios[i,j] = output_dict["cooling_ratio"]


            specific_thrusts[i,j] = output_dict["specific thrust"]
            specific_powers[i,j] = output_dict["core specific power"]
            core_powers[i,j] = output_dict["core power"]
            dT_intercoolers[i,j] = output_dict["delta T intercooler hot"]
            Tmaxs[i,j] = output_dict["T_max"]
            T_max_twozone[i,j] = output_dict["T_max_twozone"]
            T34s[i,j] = output_dict["T34"]
            T35s[i,j] = output_dict["T35"]
            T4s[i,j] = output_dict["T4"]

            hot_bypass_thrusts[i,j] = output_dict["hot bypass thrust"]
            cold_bypass_thrusts[i,j] = output_dict["cold bypass thrust"]
            core_thrusts[i,j] = output_dict["core thrust"]

            piston_fuelflow[i,j] = output_dict["piston fuelflow"]
            burner_fuelflow[i,j] = output_dict["burner fuelflow"]

            cool_ngv[i,j] = output_dict["m_cool_ngv"]
            cool_rotor[i,j] = output_dict["m_cool_rotor"]
            m_core[i,j] = output_dict["core mass flow"]

            bores[i,j] = output_dict["bore"]
            piston_bprs[i,j] = output_dict["bpr_piston"]
            piston_power_spec[i,j] = output_dict["piston_specific_power"]
            piston_powers[i,j] = output_dict["piston_power"]

            piston_heatloss[i,j] = output_dict["piston_heatloss"]
            piston_powers_indicated[i,j] = output_dict["piston_power_indicated"]
            heatloss_percentage[i,j] = output_dict["heatloss_percentage"]
            friction_percentage[i,j] = output_dict["friction_percentage"]




        lap2 = timer()
        print(f"Simulation time for 1 point: {lap2 - lap1} seconds")

        j = j + 1
    i = i + 1


end = timer()
print(f"Total simulation time for {i*j} evaluation points: {end - start} seconds")


#little bit of post processing

# engine displacement in m3
disp = 24 * bores*bores*bores*np.pi/4

# specific core power (W per m3)
core_spec_power = core_powers / disp

NO_per_power = (1000 * m_NO_tot * 3600) / (core_powers * 1e-3)

# calculate fuel split between piston engine and burner
pe_fuel_percentage = piston_fuelflow / (piston_fuelflow + burner_fuelflow)
burner_fuel_percentage = burner_fuelflow / (piston_fuelflow + burner_fuelflow)

# make params 2d
params_1 = params_1.reshape(1, -1)

# add nan to params2
params_2 = np.insert(params_2, 0, np.nan)
params_2 = params_2.reshape(1, -1)

# create arrays for saving
thermal_effs = np.concatenate((params_1.T, thermal_effs*100), axis=1)
thermal_effs = np.concatenate((params_2, thermal_effs), axis=0)

# grams per second of NOx
m_NOx_tot = np.concatenate((params_1.T, m_NO_tot*1000), axis=1)
m_NOx_tot = np.concatenate((params_2, m_NOx_tot), axis=0)

# mg nox per newton of thrust
specific_nox = np.concatenate((params_1.T, specific_nox*1e6), axis=1)
specific_nox= np.concatenate((params_2, specific_nox), axis=0)


# specific gas genarator power of gg per liter of piston
gg_spec_power = np.concatenate((params_1.T, gg_disp_spec_powers*1e-6), axis=1)
gg_spec_power = np.concatenate((params_2, gg_spec_power), axis=0)

# specific core power of gg per liter of piston
core_spec_power = np.concatenate((params_1.T, core_spec_power*1e-6), axis=1)
core_spec_power = np.concatenate((params_2, core_spec_power), axis=0)


# peak pressure do to find limits
peak_pressure = np.concatenate((params_1.T, pmaxs*1e-5), axis=1)
peak_pressure = np.concatenate((params_2, peak_pressure), axis=0)

# bore to find limits
bore = np.concatenate((params_1.T, bores*1000), axis=1)
bore = np.concatenate((params_2, bore), axis=0)

# piston outlet temp to find limits
Tout_piston = np.concatenate((params_1.T, T34s), axis=1)
Tout_piston = np.concatenate((params_2, Tout_piston), axis=0)


bprs = np.concatenate((params_1.T, bprs), axis=1)
bprs = np.concatenate((params_2, bprs), axis=0)

#fuel_consumption = np.vstack((params, SFCs*1e6)).transpose()
#bypass_ratios = np.vstack((params, bprs)).transpose()
#eff_gg = np.vstack((params, gg_effs*100)).transpose()
# bore


#pe_fuel_percentage = np.vstack((params, pe_fuel_percentage * 100)).transpose()
#burner_fuel_percentage = np.vstack((params, burner_fuel_percentage * 100)).transpose()


# save output for carpet plotting
np.savetxt(f"./results/{param_name}/thermal_eff.dat", thermal_effs, fmt="%.5f")
np.savetxt(f"./results/{param_name}/m_NOx.dat", m_NOx_tot, fmt="%.5f")
np.savetxt(f"./results/{param_name}/specific_nox.dat", specific_nox, fmt="%.5f")
np.savetxt(f"./results/{param_name}/gg_spec_power.dat", gg_spec_power, fmt="%.5f")
np.savetxt(f"./results/{param_name}/core_spec_power.dat", core_spec_power, fmt="%.5f")
np.savetxt(f"./results/{param_name}/peak_pressure.dat", peak_pressure, fmt="%.5f")
np.savetxt(f"./results/{param_name}/bore.dat", bore, fmt="%.5f")
np.savetxt(f"./results/{param_name}/Tout_piston.dat", Tout_piston, fmt="%.5f")
np.savetxt(f"./results/{param_name}/BPR.dat", bprs, fmt="%.5f")
