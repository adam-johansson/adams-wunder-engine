import numpy as np
import os
import sys
import shutil
import tempfile

sys.path.append(os.path.abspath("./../../../"))

seed = 3  # change to 2, 3 for other runs
# seed 4 is for higher peak pressure = 200 bar

# limits:
if seed in [1, 2, 3]:
    pmax_lim = 150
    T_out_lim = 1250
elif seed in [4, 5, 6]:
    pmax_lim = 200
    T_out_lim = 1250
elif seed in [7, 8, 9]:
    pmax_lim = 150
    T_out_lim = 1350
elif seed in [10, 11, 12]:
    pmax_lim = 200
    T_out_lim = 1350

print(f"pmax lim {pmax_lim}, Toutlim: {T_out_lim}")



cea_work_dir = os.path.abspath(f"optimisation_data/seed_{seed}")
os.makedirs(cea_work_dir, exist_ok=True)
os.chdir(cea_work_dir)

# CHANGED: capture this AFTER chdir, used to give each worker its own isolated temp folder
root_dir = os.path.abspath(os.getcwd())

from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib

from timeit import default_timer as timer
from multiprocessing import Pool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize

from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback
from pymoo.core.population import Population

import pandas as pd

operating_point = "TOC"

input_file = f"MR_{operating_point}_jetA_AST_optimisation"
input_dir = "CCE.input.cce_jetA"
path = input_dir + "." + input_file

input_file_pist = "4stroke_jetA"
input_dir_pist = "CCE.input.piston"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)

flags = ["life_hack", "cce"]

# These stay as module-level TEMPLATES from here on -- never mutated directly.
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

meta_model = "placeholder"

eta_p_hpc_0 = cce_input["eta_p_hpc"]
eta_lpt_0 = cce_input["eta_lpt"]

EXTRA_KEYS = [
    "thrust", "bpr", "bore", "bpr piston", "m0",
    "T_in_piston", "T_out_piston", "T35", "P max (bar)",
    "T max", "T_max_twozone", "piston_shaft_power",
    "piston_indicated_power", "piston_heatloss",
    "m_nox_pe", "m_nox_burner", "core_power",
    "core_power_per_litre", "cooling_ratio",
    "piston_fuelflow", "burner_fuelflow", "piston_fuelsplit", "error"
]


def evaluate_cce(x, root_dir):
    opr, T4, split, cr, far, pi_pe, ic_ratio, phi_sc, m_wiebe, phi_cd = x

    error = False
    lap1 = timer()
    old_cwd = os.getcwd()

    # CHANGED: isolated per-call temp directory, since CEA writes files to CWD.
    # Without this, parallel workers sharing one CWD would clobber each other's
    # CEA temp files mid-computation -- exactly the bug your colleague's code avoids.
    with tempfile.TemporaryDirectory(dir=root_dir, prefix="worker_") as run_dir:
        try:
            for lib_file in ["thermo.lib", "trans.lib"]:
                src_path = os.path.join(root_dir, lib_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(run_dir, lib_file))
            os.chdir(run_dir)

            # CHANGED: local copies, never mutate the module-level templates.
            cce_input_local = dict(cce_input)
            piston_input_local = dict(piston_input)

            cce_input_local["OPR"] = opr
            cce_input_local["T4"] = T4
            cce_input_local["PR"] = split
            cce_input_local["cr"] = cr
            cce_input_local["far piston"] = (far / 100) * (44 / 43)
            cce_input_local["pi_pe"] = pi_pe
            cce_input_local["ratio IC"] = ic_ratio
            cce_input_local["start_of_combustion"] = phi_sc
            piston_input_local["m_wiebe"] = m_wiebe
            piston_input_local["phi_cd"] = (phi_cd / 180) * np.pi

            cce_input_local["eta_p_hpc"] = eta_p_hpc_0
            cce_input_local["eta_lpt"] = eta_lpt_0

            cce_input_local["life_hack"] = "Simulate"
            cce_input_local["bpr"] = 20
            output_dict = cce_propulsion_system_specific.run_cce(
                cce_input_local, piston_input_local, flags, meta_model
            )

            if output_dict["error"]:
                error = True
            else:
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
                # CHANGED: renamed from `dict` to `bpr_result` -- your original code
                # shadowed the builtin `dict` name, which would break `dict(cce_input)` copies.
                bpr_result = auxiliaries.run_cce_bpr(cce_input_local, piston_input_local, meta_model)

                if bpr_result["error"]:
                    error = True
                else:
                    cce_input_local["bpr"] = bpr_result["bpr"][0]
                    cce_input_local["bore"] = bpr_result["bore_match"]

                    cce_input_local["life_hack"] = "Simulate_final"
                    try:
                        output_dict = cce_propulsion_system_specific.run_cce(
                            cce_input_local, piston_input_local, flags, meta_model
                        )
                    except RuntimeError:
                        error = True
                    if not error and output_dict["error"]:
                        error = True
        finally:
            os.chdir(old_cwd)

    lap2 = timer()
    print(f"Simulation time for 1 point: {lap2 - lap1} seconds")

    if error:
        specific_nox = 999
        eta_th = 0.0
        extras = {key: 0 for key in EXTRA_KEYS}
    else:
        eta_th = output_dict["thermal efficiency"]
        specific_nox = output_dict["thrust specific nox"] * 1e6

        bore = output_dict["bore"]
        bpr = bpr_result["bpr"][0]
        thrust = output_dict["thrust"]
        m0 = output_dict["mass flow"]
        T_out_piston = output_dict["T34"]
        T_in_piston = output_dict["T31"]
        pmax = output_dict["p_max"]
        bpr_piston = output_dict["bpr_piston"]
        T35 = output_dict["T35"]
        piston_shaft_power = output_dict["piston_power"]
        piston_indicated_power = output_dict["piston_power_indicated"]
        piston_heatloss = output_dict["piston_heatloss"]
        m_nox_pe = output_dict["m_nox_PE"]
        m_nox_burner = output_dict["m_nox_burner"]
        T_max = output_dict["T_max"]
        T_max_twozone = output_dict["T_max_twozone"]
        displacement = output_dict["engine displacement"]
        cooling_ratio = output_dict["cooling_ratio"]
        core_power = output_dict["core power"]
        core_power_per_litre = (core_power / displacement) * 1e-6
        piston_fuelflow = output_dict["piston fuelflow"]
        burner_fuelflow = output_dict["burner fuelflow"]
        piston_fuelsplit = piston_fuelflow / (piston_fuelflow + burner_fuelflow)

        extras = {
            "thrust": thrust, "bpr": bpr, "bore": bore, "bpr piston": bpr_piston,
            "m0": m0, "T_in_piston": T_in_piston, "T_out_piston": T_out_piston,
            "T35": T35, "P max (bar)": pmax * 1e-5, "T max": T_max,
            "T_max_twozone": T_max_twozone, "piston_shaft_power": piston_shaft_power,
            "piston_indicated_power": piston_indicated_power, "piston_heatloss": piston_heatloss,
            "m_nox_pe": m_nox_pe, "m_nox_burner": m_nox_burner, "core_power": core_power,
            "core_power_per_litre": core_power_per_litre, "cooling_ratio": cooling_ratio,
            "piston_fuelflow": piston_fuelflow, "burner_fuelflow": burner_fuelflow,
            "piston_fuelsplit": piston_fuelsplit, "error": error,
        }

        print(f'opr: {opr}, T4: {T4}, split: {split}, cr: {cr}, far: {far}, pi_pe: {pi_pe}, ic ratio: {ic_ratio}')
        print(f"start of combustion: {phi_sc}, m_wiebe: {m_wiebe}, combustion duration: {phi_cd}")
        print(f"Point converged and: thermal efficiency {eta_th*100} % and specific nox: {specific_nox} mg/Ns")

    objectives = np.array([-eta_th, specific_nox])
    return objectives, extras


# CHANGED: ElementwiseProblem -- pymoo calls _evaluate once per individual and
# distributes them across workers via elementwise_runner, instead of your
# original manual for-loop over the whole population.
class MyEngineProblem(ElementwiseProblem):
    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir
        super().__init__(
            n_var=10,
            n_obj=2,
            n_constr=5,
            xl=np.array([10, 1000, 0.0, 4, 2, 0.9, 0.0, 340, 0.5, 20]),
            xu=np.array([30, 1600, 0.5, 15, 5, 1.8, 1.0, 380, 5.0, 80]),
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        obj, extra = evaluate_cce(x, self.root_dir)

        constraints = [
            extra["T_out_piston"] - T_out_lim,
            extra["P max (bar)"] - pmax_lim,
            extra["bore"] - 0.2,
            -extra["bpr piston"],
            1 if extra["error"] else -1,  # failed simulation -> infeasible
        ]

        out["F"] = obj
        out["G"] = np.array(constraints)
        # CHANGED: stash extras here instead of appending to a global list --
        # a global `all_evaluations.append(...)` would only update that ONE
        # worker process's private memory, invisible to the main process.
        out["extra"] = extra


class OptimisationCallback(Callback):
    def __init__(self, ref_point, seed, gen_history=None, hv_history=None, all_evaluations=None):
        super().__init__()
        self.ref_point = ref_point
        self.seed = seed
        self.hv_history = hv_history or []
        self.gen_history = gen_history or []
        self.gen_offset = gen_history[-1] if gen_history else 0
        # CHANGED: all_evaluations now lives on the callback (main-process only),
        # instead of a module-level global that workers can't safely share.
        self.all_evaluations = all_evaluations or []

    def notify(self, algorithm):
        gen = algorithm.n_gen + self.gen_offset
        output_dir = "."

        # CHANGED: pull design vars, objectives, and stashed extras back from
        # the population -- this is where per-individual results collected
        # across worker processes get reassembled in the main process.
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        extras = algorithm.pop.get("extra")

        var_names = ["opr", "T4", "split", "cr", "far", "p_ratio", "IC_ratio", "phi_sc", "m_wiebe", "phi_cd"]

        for xi, fi, exi in zip(X, F, extras):
            record = {name: val for name, val in zip(var_names, xi)}
            record["eta_th"] = -fi[0]
            record["specific_nox"] = fi[1]
            record.update(exi)
            record["constraint_violation"] = (
                max(0, exi["T_out_piston"] - T_out_lim)
                + max(0, exi["P max (bar)"] - pmax_lim)
                + max(0, exi["bore"] - 0.2)
                + max(0, -exi["bpr piston"])
                + max(0, 1 if exi["error"] else -1)
            )
            record["is_feasible"] = (
                exi["T_out_piston"] <= T_out_lim
                and exi["P max (bar)"] <= pmax_lim
                and exi["bore"] <= 0.2
                and exi["bpr piston"] > 0
                and not exi["error"]
            )
            self.all_evaluations.append(record)

        all_df = pd.DataFrame(self.all_evaluations)
        all_df.to_csv(f"{output_dir}/all_evaluations.csv", index=False)

        if algorithm.opt is not None:
            pareto_df = pd.DataFrame(
                np.hstack([algorithm.opt.get("X"), algorithm.opt.get("F")]),
                columns=var_names + ['eta_th', 'specific_nox']
            )
            pareto_df['eta_th'] = -pareto_df['eta_th']
            pareto_df.to_csv(f"{output_dir}/pareto_solutions.csv", index=False)

        hv_indicator = HV(ref_point=self.ref_point)
        hv = hv_indicator(algorithm.opt.get("F"))
        self.hv_history.append(hv)
        self.gen_history.append(gen)

        hv_df = pd.DataFrame({'generation': self.gen_history, 'hypervolume': self.hv_history})
        hv_df.to_csv(f"{output_dir}/hypervolume.csv", index=False)

        pareto_size = len(algorithm.opt)
        feasible_total = sum(all_df['is_feasible']) if 'is_feasible' in all_df.columns else '?'
        print(f"Gen {gen:3d} | HV: {hv:.4f} | Pareto size: {pareto_size:3d} | "
              f"Total evals: {len(self.all_evaluations)} | Feasible: {feasible_total} ")


# CHANGED: everything below MUST be inside this guard on Windows. Without it,
# every spawned worker process would re-import this file fresh and re-run
# minimize(...) itself, causing runaway recursive process spawning.
if __name__ == "__main__":

    resume_optimisation = False
    n_gen = 5
    new_gens = 5
    pop_size = 150
    output_dir = "."
    
    all_evaluations = []
    if resume_optimisation and os.path.exists(f"{output_dir}/all_evaluations.csv"):
        existing_df = pd.read_csv(f"{output_dir}/all_evaluations.csv")
        all_evaluations = existing_df.to_dict('records')

    hv_csv_path = f"{output_dir}/hypervolume.csv"
    if resume_optimisation:
        hv_df_existing = pd.read_csv(hv_csv_path)
        existing_gen = hv_df_existing['generation'].tolist()
        existing_hv = hv_df_existing['hypervolume'].tolist()
    else:
        existing_gen = []
        existing_hv = []

    ref_point = np.array([-0.35, 1.5])

    callback = OptimisationCallback(
        ref_point=ref_point,
        seed=seed,
        gen_history=existing_gen,
        hv_history=existing_hv,
        all_evaluations=all_evaluations,
    )

    # CHANGED: this is the actual parallelization -- a process pool feeding
    # pymoo's per-individual evaluations through StarmapParallelization.
    n_processes = max(os.cpu_count() - 2, 2)
    print(f"Spawning pool with {n_processes} worker processes...")
    pool = Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)

    problem = MyEngineProblem(root_dir=root_dir, elementwise_runner=runner)
    algorithm = NSGA2(pop_size=pop_size) #testa variera sen

    print(algorithm.mating.crossover.eta.value)
    print(algorithm.mating.mutation.eta.value)
    print(algorithm.mating.crossover.prob.value)
    print(algorithm.mating.mutation.prob.value)

    t_start = timer()

    if not resume_optimisation:
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=True, callback=callback)
    else:
        pop_X = np.loadtxt(f"{output_dir}/last_population_X.csv", delimiter=",")
        pop_F = np.loadtxt(f"{output_dir}/last_population_F.csv", delimiter=",")
        gen_done = int(np.loadtxt(f"{output_dir}/last_generation.csv", delimiter=","))
        pop = Population.new("X", pop_X, "F", pop_F)
        algorithm.initialization.sampling = pop
        res = minimize(problem, algorithm, ('n_gen', new_gens), seed=seed, verbose=True, callback=callback)

    t_end = timer()
    t_total = t_end - t_start

    pool.close()
    pool.join()

    # --- final saving, same as before ---
    pop_X = res.pop.get("X")
    pop_F = res.pop.get("F")
    np.savetxt(f"{output_dir}/last_population_X.csv", pop_X, delimiter=",")
    np.savetxt(f"{output_dir}/last_population_F.csv", pop_F, delimiter=",")
    gen_offset = gen_done if resume_optimisation else 0
    np.savetxt(f"{output_dir}/last_generation.csv", [res.algorithm.n_gen + gen_offset], delimiter=",")

    var_names = ["opr", "T4", "split", "cr", "far", "p_ratio", "IC_ratio", "phi_sc", "m_wiebe", "phi_cd"]
    pareto_df = pd.DataFrame(
        np.hstack([res.X, res.F]),
        columns=var_names + ['eta_th', 'specific_nox']
    )
    pareto_df['eta_th'] = -pareto_df['eta_th']
    pareto_df.to_csv(f"{output_dir}/pareto_solutions.csv", index=False)

    hours = int(t_total // 3600)
    minutes = int((t_total % 3600) // 60)
    seconds = t_total % 60
    print(f"\nOptimization Summary:")
    print(f"Total run time: {hours}h {minutes}m {seconds:.1f}s")
    print(f"Total evaluations: {len(callback.all_evaluations)}")
    print(f"Pareto-optimal solutions: {len(pareto_df)}")