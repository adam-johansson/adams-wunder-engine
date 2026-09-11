import sys
sys.path.append("./../../..")


from CCE.src import geared_turbofan_jetA 

import importlib
import numpy as np

from scipy.optimize import fsolve


from timeit import default_timer as timer



input_file = "mid_range_TOC_jetA_conventional"
input_dir = "CCE.input.conventional_jetA"
path = input_dir + "." + input_file
d = importlib.import_module(path)


data_dict = {
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
    "eta_hpt": d.eta_hpt,
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
    "dp_rec": d.dp_rec,
    "dT_rec": d.dT_rec,
    "HPT_eff_type": d.HPT_eff_type,
    "LPT_eff_type": d.LPT_eff_type,
}


flags = []

F_goal = 27.180417543091878*1e3

#data_dict["OPR"] = 50
#data_dict["T4"] = 1900

def find_bpr(x):

    data_dict["bpr"] = x[0]

    output  = geared_turbofan_jetA.run_turbofan(data_dict, flags)

    F = output["thrust"]
    residual = np.array([F - F_goal])
    print(f"Thrust residual {residual} bpr {x}")
    return residual

bpr, info, ier, msg = fsolve(find_bpr, x0=11, full_output=True)

print(bpr)

data_dict["bpr"] = bpr[0]

output  = geared_turbofan_jetA.run_turbofan(data_dict, flags)


sfc = output["sfc"]
m0 = output["m0"]
F = output["thrust"]
vel_ratio = output["vel_ratio"]
F_nox = output["thrust_nox"]
eta_th = output["eta_th"]


print(f"mass flow: {m0} [kg/s]")
print(f"SFC: {sfc*1e6} [mg/Ns]")
print(f"Thrust: {F*1e-3} [kN]]")
print(f"Velocity ratio: {vel_ratio}")
print(f"Thrust specific NOx: {F_nox*1e6} mg/Ns")
print(f"Thermal efficiency: {eta_th} ")