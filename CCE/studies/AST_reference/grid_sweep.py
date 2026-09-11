import sys
sys.path.append("./../../..")

import numpy as np
import pandas as pd
import importlib
from scipy.optimize import fsolve

from CCE.src import geared_turbofan_jetA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


input_file = "mid_range_TOC_jetA_conventional"
input_dir = "CCE.input.conventional_jetA"
path = input_dir + "." + input_file
d = importlib.import_module(path)

data_dict = {
    "Fn": d.Fn, "dTisa": d.dTisa, "bpr": d.bpr, "T4": d.T4,
    "fpr_outer": d.fpr_outer, "Fs_req": d.Fs_req, "dp_intake": d.dp_intake,
    "dp_bypass": d.dp_bypass, "M": d.M, "eta_fan": d.eta_fan,
    "eta_p_hpc": d.eta_p_hpc, "eta_p_lpc": d.eta_p_lpc, "eta_b": d.eta_b,
    "dPcomb": d.dPcomb, "eta_s": d.eta_s, "eta_g": d.eta_g,
    "eta_hpt": d.eta_hpt, "eta_lpt": d.eta_lpt, "cfg_core": d.cfg_core,
    "cfg_bypass": d.cfg_bypass, "cd_nozzle": d.cd_nozzle, "alt": d.alt,
    "fuel": d.fuel, "OPR": d.OPR, "PR": d.PR, "t_fuel": d.t_fuel,
    "t_tank": d.t_tank, "power_offtake": d.power_offtake, "dp_rec": d.dp_rec,
    "dT_rec": d.dT_rec, "HPT_eff_type": d.HPT_eff_type, "LPT_eff_type": d.LPT_eff_type,
}

flags = []
F_goal = 27.180417543091878 * 1e3


def evaluate_point(opr, T4):
    data_dict["OPR"] = opr
    data_dict["T4"] = T4

    def find_bpr(x):
        data_dict["bpr"] = x[0]
        output = geared_turbofan_jetA.run_turbofan(data_dict, flags)
        return np.array([output["thrust"] - F_goal])

    bpr, info, ier, msg = fsolve(find_bpr, x0=11, full_output=True)
    data_dict["bpr"] = bpr[0]

    output = geared_turbofan_jetA.run_turbofan(data_dict, flags)

    eta_th = output["eta_th"]
    specific_nox = output["thrust_nox"] * 1e6

    return eta_th, specific_nox


# --- Grid sweep ---
opr_values = np.linspace(30, 60, 40)   # adjust range/resolution as needed
T4_values = np.linspace(1600, 2000, 40)

all_evaluations = []

for opr in opr_values:
    for T4 in T4_values:
        eta_th, specific_nox = evaluate_point(opr, T4)
        all_evaluations.append({
            "opr": opr,
            "T4": T4,
            "eta_th": eta_th,
            "specific_nox": specific_nox,
        })
        print(f"OPR={opr:.2f}, T4={T4:.1f} -> eta_th={eta_th*100:.2f}%, NOx={specific_nox:.3f}")

all_df = pd.DataFrame(all_evaluations)
all_df.to_csv("all_evaluations.csv", index=False)

# --- Extract Pareto front (maximize eta_th, minimize specific_nox) ---
F = np.column_stack([-all_df["eta_th"].values, all_df["specific_nox"].values])
front_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)

pareto_df = all_df.iloc[front_idx].sort_values("eta_th")
pareto_df.to_csv("pareto_solutions.csv", index=False)