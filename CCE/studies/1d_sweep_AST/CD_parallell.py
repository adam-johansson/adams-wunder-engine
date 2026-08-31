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

operating_point = "TOC"

input_file = f"MR_{operating_point}_jetA_AST_baseline"
input_dir = "CCE.input.cce_jetA"
path = input_dir + "." + input_file

input_file_pist = "4stroke_jetA"
input_dir_pist = "CCE.input.piston"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)

flags = ["life_hack", "cce"]

constant_F = False

cce_input = {
    "Fn": d.Fn, "dTisa": d.dTisa, "bpr": d.bpr, "T4": d.T4, "fpr_outer": d.fpr_outer,
    "Fs_req": d.Fs_req, "dp_intake": d.dp_intake, "dp_bypass": d.dp_bypass, "M": d.M,
    "eta_fan": d.eta_fan, "eta_p_hpc": d.eta_p_hpc, "eta_p_lpc": d.eta_p_lpc,
    "eta_b": d.eta_b, "dPcomb": d.dPcomb, "eta_s": d.eta_s, "eta_g": d.eta_g,
    "eta_lpt": d.eta_lpt, "cfg_core": d.cfg_core, "cfg_bypass": d.cfg_bypass,
    "cd_nozzle": d.cd_nozzle, "alt": d.alt, "fuel": d.fuel, "OPR": d.OPR, "PR": d.PR,
    "t_fuel": d.t_fuel, "t_tank": d.t_tank, "power_offtake": d.power_offtake,
    "surrogate": d.surrogate, "second_burner": d.second_burner, "pi_pe": d.pi_pe,
    "cr": d.cr, "bore": d.bore, "far piston": d.far_piston,
    'effectiveness IC': d.eff_IC, 'dp_inter_compressor': d.dp_inter_compressor,
    "intercooler": d.intercooler, "specific": d.specific, "v_mean": d.v_mean,
    "start_of_combustion": d.start_of_combustion, "ratio IC": d.ratio_IC,
    "piston_mode": d.piston_mode, "LPT_eff_type": d.LPT_eff_type,
    "EGR_rate": d.EGR_rate, "oil_temp": d.oil_temp,
}

piston_input = {
    'p_in': d_p.p_in, 'T_in': d_p.T_in, 'p_ratio': d_p.p_ratio, 'cycle': d_p.cycle,
    'cooling': d_p.cooling, 'opposed': d_p.opposed, 'cr': d_p.cr, 'bore': d_p.d,
    'bsr': d_p.bsr, 'v_mean': d_p.v_mean, 'lms': d_p.lms, 'Twalls': d_p.Twalls,
    'ch': d_p.ch, 'valve_timings': d_p.valve_timings, 'n_valve': d_p.n_valve,
    'lv_max': d_p.lv_max, 'cd': d_p.cd, 'eta_c': d_p.eta_c, 'mf_tot': d_p.mf_tot,
    'wa': d_p.wa, 'wm': d_p.wm, 'm_wiebe': d_p.m_wiebe, 'phi_sc': d_p.phi_sc,
    'phi_cd': d_p.phi_cd, 'T_fuel': d_p.T_fuel, 'p_fuel': d_p.p_fuel, 'it': d_p.it,
    'wiebe_type': d_p.wiebe_type, 'valve_type': d_p.valve_type,
    'far_goal': d_p.far_goal, 'cylinders': d_p.cylinders, 'fuel': d_p.fuel,
    'c1': d_p.c1, 'c4': d_p.c4, 'c5': d_p.c5, 'premixed': d_p.premixed,
}

param_name = "CD"  # start of combustion only
params_1 = np.arange(20, 100.1, 5.0)  # phi_sc sweep -- everything else fixed

num1 = np.size(params_1)

SFCs = np.zeros(num1)
EI_noxs = np.zeros(num1)
m_noxs_pe = np.zeros(num1)
m_noxs_burner = np.zeros(num1)
m_NO_tot = np.zeros(num1)
NO_ppm_piston = np.zeros(num1)
specific_nox = np.zeros(num1)
pmaxs = np.zeros(num1)
dT_intercoolers = np.zeros(num1)
Tmaxs = np.zeros(num1)
T_max_twozone = np.zeros(num1)
T4s = np.zeros(num1)
T34s = np.zeros(num1)
T35s = np.zeros(num1)
core_effs = np.zeros(num1)
thermal_effs = np.zeros(num1)
transmission_effs = np.zeros(num1)
propulsive_effs = np.zeros(num1)
overall_effs = np.zeros(num1)
gg_effs = np.zeros(num1)
specific_thrusts = np.zeros(num1)
specific_powers = np.zeros(num1)
core_powers = np.zeros(num1)
gg_powers = np.zeros(num1)
gg_mass_spec_powers = np.zeros(num1)
gg_disp_spec_powers = np.zeros(num1)
cooling_ratios = np.zeros(num1)
hot_bypass_thrusts = np.zeros(num1)
cold_bypass_thrusts = np.zeros(num1)
core_thrusts = np.zeros(num1)
piston_fuelflow = np.zeros(num1)
burner_fuelflow = np.zeros(num1)
cool_ngv = np.zeros(num1)
cool_rotor = np.zeros(num1)
m_core = np.zeros(num1)
bprs = np.zeros(num1)
bores = np.zeros(num1)
piston_bprs = np.zeros(num1)
piston_power_spec = np.zeros(num1)
piston_powers = np.zeros(num1)
piston_heatloss = np.zeros(num1)
piston_powers_indicated = np.zeros(num1)
heatloss_percentage = np.zeros(num1)
friction_percentage = np.zeros(num1)

meta_model = "placeholder"

eta_p_hpc_0 = cce_input["eta_p_hpc"]
eta_lpt_0 = cce_input["eta_lpt"]


def evaluate_grid_point(i, phi_cd, root_dir):
    lap1 = timer()
    result = _evaluate_grid_point_inner(i, phi_cd, root_dir)
    lap2 = timer()
    status = "FAILED" if result["error"] else "ok"
    print(f"[{status}] phi_sc={phi_cd:.2f}  (point {i})  --  {lap2 - lap1:.2f} s")
    return result


def _evaluate_grid_point_inner(i, phi_cd, root_dir):
    old_cwd = os.getcwd()
    result = {"i": i, "error": False}

    with tempfile.TemporaryDirectory(dir=root_dir, prefix="worker_") as run_dir:
        try:
            for lib_file in ["thermo.lib", "trans.lib"]:
                src_path = os.path.join(root_dir, lib_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(run_dir, lib_file))
            os.chdir(run_dir)

            cce_input_local = dict(cce_input)
            piston_input_local = dict(piston_input)


            piston_input_local["phi_cd"] = (phi_cd/180)*np.pi


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
                "heatloss_percentage": "heatloss_percentage", "friction_percentage": "friction_percentage", "NO ppm piston": "NO ppm piston",
            }
            for result_key, output_key in key_map.items():
                result[result_key] = output_dict[output_key]
        finally:
            os.chdir(old_cwd)

    return result


if __name__ == "__main__":
    root_dir = os.path.abspath(os.getcwd())

    n_processes = max(os.cpu_count() - 2, 2)
    print(f"Spawning pool with {n_processes} worker processes...")

    tasks = [(i, phi_sc, root_dir) for i, phi_sc in enumerate(params_1)]

    start = timer()
    with Pool(n_processes) as pool:
        results = pool.starmap(evaluate_grid_point, tasks)
    end = timer()
    print(f"Total simulation time for {len(tasks)} evaluation points: {end - start} seconds")

    for r in results:
        i = r["i"]
        if r["error"]:
            bprs[i] = r.get("bpr_fallback", 0)
            continue
        SFCs[i] = r["SFC"]
        pmaxs[i] = r["pmax"]
        EI_noxs[i] = r["EI_nox"]
        m_noxs_pe[i] = r["m_nox_pe"]
        m_noxs_burner[i] = r["m_nox_burner"]
        m_NO_tot[i] = r["m_NO_tot"]
        NO_ppm_piston[i] = r["NO ppm piston"]
        specific_nox[i] = r["specific_nox"]
        core_effs[i] = r["core_eff"]
        transmission_effs[i] = r["transmission_eff"]
        thermal_effs[i] = r["thermal_eff"]
        propulsive_effs[i] = r["propulsive_eff"]
        overall_effs[i] = r["overall_eff"]
        gg_effs[i] = r["gg_eff"]
        gg_powers[i] = r["gg_power"]
        gg_mass_spec_powers[i] = r["gg_mass_spec_power"]
        gg_disp_spec_powers[i] = r["gg_disp_spec_power"]
        cooling_ratios[i] = r["cooling_ratio"]
        specific_thrusts[i] = r["specific_thrust"]
        specific_powers[i] = r["specific_power"]
        core_powers[i] = r["core_power"]
        dT_intercoolers[i] = r["dT_intercooler"]
        Tmaxs[i] = r["Tmax"]
        T_max_twozone[i] = r["T_max_twozone"]
        T34s[i] = r["T34"]
        T35s[i] = r["T35"]
        T4s[i] = r["T4_out"]
        hot_bypass_thrusts[i] = r["hot_bypass_thrust"]
        cold_bypass_thrusts[i] = r["cold_bypass_thrust"]
        core_thrusts[i] = r["core_thrust"]
        piston_fuelflow[i] = r["piston_fuelflow"]
        burner_fuelflow[i] = r["burner_fuelflow"]
        cool_ngv[i] = r["cool_ngv"]
        cool_rotor[i] = r["cool_rotor"]
        m_core[i] = r["m_core"]
        bores[i] = r["bore"]
        piston_bprs[i] = r["piston_bpr"]
        piston_power_spec[i] = r["piston_power_spec"]
        piston_powers[i] = r["piston_power"]
        piston_heatloss[i] = r["piston_heatloss"]
        piston_powers_indicated[i] = r["piston_power_indicated"]
        heatloss_percentage[i] = r["heatloss_percentage"]
        friction_percentage[i] = r["friction_percentage"]
        bprs[i] = r["bpr"]

    # --- post processing ---
    disp = 24 * bores*bores*bores*np.pi/4
    core_spec_power = core_powers / disp
    NO_per_power = (1000 * m_NO_tot * 3600) / (core_powers * 1e-3)
    pe_fuel_percentage = piston_fuelflow / (piston_fuelflow + burner_fuelflow)
    burner_fuel_percentage = burner_fuelflow / (piston_fuelflow + burner_fuelflow)

    # simple two-column (phi_sc, value) tables -- no carpet-grid header needed for a 1D sweep
    os.makedirs(f"./results/{param_name}", exist_ok=True)

    def save_1d(filename, values):
        table = np.column_stack((params_1, values))
        np.savetxt(f"./results/{param_name}/{filename}", table, fmt="%.5f",
                   header="phi_sc  value", comments="")


    save_1d("thermal_eff.dat", thermal_effs*100)
    save_1d("m_NOx.dat", m_NO_tot*1000)
    save_1d("specific_nox.dat", specific_nox*1e6)
    save_1d("gg_spec_power.dat", gg_disp_spec_powers*1e-6)
    save_1d("core_spec_power.dat", core_spec_power*1e-6)
    save_1d("peak_pressure.dat", pmaxs*1e-5)
    save_1d("Tmax.dat", Tmaxs)
    save_1d("Tmax2zone.dat", T_max_twozone)
    save_1d("T34.dat", T34s)
    save_1d("T35.dat", T35s)
    save_1d("NO_ppm_piston.dat", NO_ppm_piston)
    save_1d("bore.dat", bores*1000)
    save_1d("Tout_piston.dat", T34s)
    save_1d("BPR.dat", bprs)
    save_1d("piston_BPRS.dat", piston_bprs)