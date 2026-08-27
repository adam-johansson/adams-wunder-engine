import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import shutil

from multiprocessing import Pool

import sys
sys.path.append("./../../../")




from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib

from timeit import default_timer as timer

# Importing input parameters


operating_point = "TOC"

input_file = f"MR_{operating_point}_jetA_AST_baseline"
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
    "EGR_rate": d.EGR_rate,
    "oil_temp": d.oil_temp,
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




param_name = "OPR_SC"
params_1 = np.arange(16,26.1,2)
params_2 = np.arange(340,380.1,5)

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

def evaluate_grid_point(i, j, opr, phi_sc, root_dir):
    lap1 = timer()
    result = _evaluate_grid_point_inner(i, j, opr, phi_sc, root_dir)
    lap2 = timer()
    status = "FAILED" if result["error"] else "ok"
    print(f"[{status}] OPR={opr:.2f}, phi_sc={phi_sc:.1f}  (point {i},{j})  --  {lap2 - lap1:.2f} s")
    return result


def _evaluate_grid_point_inner(i, j, opr, phi_sc, root_dir):
    old_cwd = os.getcwd()
    result = {"i": i, "j": j, "error": False}

    with tempfile.TemporaryDirectory(dir=root_dir, prefix="worker_") as run_dir:
        try:
            for lib_file in ["thermo.lib", "trans.lib"]:
                src_path = os.path.join(root_dir, lib_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(run_dir, lib_file))
            os.chdir(run_dir)

            cce_input_local = dict(cce_input)
            piston_input_local = dict(piston_input)

            cce_input_local["OPR"] = opr
            cce_input_local["start_of_combustion"] = phi_sc
            cce_input_local["eta_p_hpc"] = eta_p_hpc_0
            cce_input_local["eta_lpt"] = eta_lpt_0

            cce_input_local["life_hack"] = "Simulate"
            cce_input_local["bpr"] = 20
            output_dict = cce_propulsion_system_specific.run_cce(
                cce_input_local, piston_input_local, flags, meta_model
            )

            if output_dict["error"]:
                result["error"] = True
                return result

            piston_input_local["k_m"] = output_dict["k_m"]
            piston_input_local["k0_T"] = output_dict["k0_T"]
            piston_input_local["k1_T"] = output_dict["k1_T"]
            piston_input_local["k0_H"] = output_dict["k0_H"]
            piston_input_local["k1_H"] = output_dict["k1_H"]
            piston_input_local["piston_specific_power"] = output_dict["piston_specific_power"]
            cce_input_local["eta_p_hpc"] = output_dict["eta_hpc"]
            cce_input_local["eta_lpt"] = output_dict["eta_lpt"]

            cce_input_local["life_hack"] = "Express"
            cce_input_local["trade_factors"] = False
            bpr_result = auxiliaries.run_cce_bpr(cce_input_local, piston_input_local, meta_model)

            if bpr_result["error"]:
                result["error"] = True
                result["bpr_fallback"] = bpr_result["bpr"]
                return result

            cce_input_local["bpr"] = bpr_result["bpr"][0]
            cce_input_local["bore"] = bpr_result["bore_match"]
            result["bpr"] = bpr_result["bpr"][0]

            cce_input_local["life_hack"] = "Simulate_final"
            output_dict = cce_propulsion_system_specific.run_cce(
                cce_input_local, piston_input_local, flags, meta_model
            )

            if output_dict["error"]:
                result["error"] = True
                return result

            key_map = {
                "SFC": "sfc", "pmax": "p_max", "EI_nox": "EI_nox",
                "m_nox_pe": "m_nox_PE", "m_nox_burner": "m_nox_burner",
                "m_NO_tot": "m_NO_tot", "specific_nox": "thrust specific nox",
                "core_eff": "core efficiency", "transmission_eff": "transmission efficiency",
                "thermal_eff": "thermal efficiency", "propulsive_eff": "propulsive efficiency",
                "overall_eff": "overall efficiency", "gg_eff": "gg efficiency",
                "gg_power": "gg_power", "gg_mass_spec_power": "gg_mass_specific_power",
                "gg_disp_spec_power": "gg_disp_specific_power", "cooling_ratio": "cooling_ratio",
                "specific_thrust": "specific thrust", "specific_power": "core specific power",
                "core_power": "core power", "dT_intercooler": "delta T intercooler hot",
                "Tmax": "T_max", "T_max_twozone": "T_max_twozone", "T34": "T34",
                "T35": "T35", "T4_out": "T4", "hot_bypass_thrust": "hot bypass thrust",
                "cold_bypass_thrust": "cold bypass thrust", "core_thrust": "core thrust",
                "piston_fuelflow": "piston fuelflow", "burner_fuelflow": "burner fuelflow",
                "cool_ngv": "m_cool_ngv", "cool_rotor": "m_cool_rotor",
                "m_core": "core mass flow", "bore": "bore", "piston_bpr": "bpr_piston",
                "piston_power_spec": "piston_specific_power", "piston_power": "piston_power",
                "piston_heatloss": "piston_heatloss", "piston_power_indicated": "piston_power_indicated",
                "heatloss_percentage": "heatloss_percentage", "friction_percentage": "friction_percentage",
            }
            for result_key, output_key in key_map.items():
                result[result_key] = output_dict[output_key]
        finally:
            os.chdir(old_cwd)

    return result

if __name__ == "__main__":
    root_dir = os.path.abspath(os.getcwd())  # add this near the top, before any chdir

    n_processes = max(os.cpu_count() - 2, 2)
    print(f"Spawning pool with {n_processes} worker processes...")

    tasks = [
        (i, j, opr, T4, root_dir)
        for i, opr in enumerate(params_1)
        for j, T4 in enumerate(params_2)
    ]

    start = timer()
    with Pool(n_processes) as pool:
        results = pool.starmap(evaluate_grid_point, tasks)
    end = timer()
    print(f"Total simulation time for {len(tasks)} evaluation points: {end - start} seconds")

    for r in results:
        i, j = r["i"], r["j"]
        if r["error"]:
            bprs[i, j] = r.get("bpr_fallback", 0)
            continue
        SFCs[i, j] = r["SFC"]
        pmaxs[i, j] = r["pmax"]
        EI_noxs[i, j] = r["EI_nox"]
        m_noxs_pe[i, j] = r["m_nox_pe"]
        m_noxs_burner[i, j] = r["m_nox_burner"]
        m_NO_tot[i, j] = r["m_NO_tot"]
        specific_nox[i, j] = r["specific_nox"]
        core_effs[i, j] = r["core_eff"]
        transmission_effs[i, j] = r["transmission_eff"]
        thermal_effs[i, j] = r["thermal_eff"]
        propulsive_effs[i, j] = r["propulsive_eff"]
        overall_effs[i, j] = r["overall_eff"]
        gg_effs[i, j] = r["gg_eff"]
        gg_powers[i, j] = r["gg_power"]
        gg_mass_spec_powers[i, j] = r["gg_mass_spec_power"]
        gg_disp_spec_powers[i, j] = r["gg_disp_spec_power"]
        cooling_ratios[i, j] = r["cooling_ratio"]
        specific_thrusts[i, j] = r["specific_thrust"]
        specific_powers[i, j] = r["specific_power"]
        core_powers[i, j] = r["core_power"]
        dT_intercoolers[i, j] = r["dT_intercooler"]
        Tmaxs[i, j] = r["Tmax"]
        T_max_twozone[i, j] = r["T_max_twozone"]
        T34s[i, j] = r["T34"]
        T35s[i, j] = r["T35"]
        T4s[i, j] = r["T4_out"]
        hot_bypass_thrusts[i, j] = r["hot_bypass_thrust"]
        cold_bypass_thrusts[i, j] = r["cold_bypass_thrust"]
        core_thrusts[i, j] = r["core_thrust"]
        piston_fuelflow[i, j] = r["piston_fuelflow"]
        burner_fuelflow[i, j] = r["burner_fuelflow"]
        cool_ngv[i, j] = r["cool_ngv"]
        cool_rotor[i, j] = r["cool_rotor"]
        m_core[i, j] = r["m_core"]
        bores[i, j] = r["bore"]
        piston_bprs[i, j] = r["piston_bpr"]
        piston_power_spec[i, j] = r["piston_power_spec"]
        piston_powers[i, j] = r["piston_power"]
        piston_heatloss[i, j] = r["piston_heatloss"]
        piston_powers_indicated[i, j] = r["piston_power_indicated"]
        heatloss_percentage[i, j] = r["heatloss_percentage"]
        friction_percentage[i, j] = r["friction_percentage"]
        bprs[i, j] = r["bpr"]

    # everything from "# little bit of post processing" onward stays exactly as you had it

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
