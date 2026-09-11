import sys
sys.path.append("./../../..")

import numpy as np
import pandas as pd
import importlib
from scipy.optimize import fsolve

from CCE.src import geared_turbofan_jetA

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize


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

all_evaluations = []


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


class TurbofanProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2,
                          xl=np.array([50, 1700]),   # OPR, T4 lower bounds
                          xu=np.array([60, 1800]))   # OPR, T4 upper bounds

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for opr, T4 in x:
            eta_th, specific_nox = evaluate_point(opr, T4)
            F.append([-eta_th, specific_nox])
            all_evaluations.append({
                "eta_th": eta_th,
                "specific_nox": specific_nox,
            })
        out["F"] = np.array(F)


class HVCallback(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.ref_point = ref_point
        self.gen_history = []
        self.hv_history = []

    def notify(self, algorithm):
        hv = HV(ref_point=self.ref_point)(algorithm.opt.get("F"))
        self.gen_history.append(algorithm.n_gen)
        self.hv_history.append(hv)
        print(f"Gen {algorithm.n_gen:3d} | HV: {hv:.4f}")


ref_point = np.array([-0.4, 1.5])  # adjust based on this cycle's expected eta_th/nox range
callback = HVCallback(ref_point)

problem = TurbofanProblem()
algorithm = NSGA2(pop_size=40)

res = minimize(problem, algorithm, ('n_gen', 30), seed=1, verbose=True, callback=callback)

pd.DataFrame(all_evaluations).to_csv("all_evaluations.csv", index=False)

pareto_df = pd.DataFrame(res.F, columns=["eta_th", "specific_nox"])
pareto_df["eta_th"] = -pareto_df["eta_th"]
pareto_df.to_csv("pareto_solutions.csv", index=False)

pd.DataFrame({
    "generation": callback.gen_history,
    "hypervolume": callback.hv_history,
}).to_csv("hypervolume.csv", index=False)