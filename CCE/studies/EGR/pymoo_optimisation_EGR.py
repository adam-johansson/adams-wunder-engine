import numpy as np

import os

import sys
#sys.path.append("./../../../")

# --- Path setup first ---
sys.path.append(os.path.abspath("./../../../"))

# Change to seed 1, 2 or 3 folder to be able to run in parallell
# seed 1,2,,3 is for secondary burner
# seed 7, 8, 9 for no secondary burner
seed = 1  # change to 2, 3 for other runs

cea_work_dir = os.path.abspath(f"optimisation_data/seed_{seed}")
#os.makedirs(cea_work_dir, exist_ok=True)
os.chdir(cea_work_dir)


from CCE.src import cce_propulsion_system_specific
from CCE.src import auxiliaries
import importlib

from timeit import default_timer as timer
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback

from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling

import pandas as pd

# Importing input parameters

operating_point = "TOC"

input_file = f"MR_{operating_point}_jetA_EGR"
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
    "trade_factors": d.trade_factors,
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


meta_model = "placeholder"
 

# save orignal efficiency values
eta_p_hpc_0 = cce_input["eta_p_hpc"] 
eta_lpt_0 = cce_input["eta_lpt"]

# Global storage for all evaluated points
all_evaluations = []

# function for evaluating the CCE system

def evaluate_cce(x):

    #print(f'opr: {x[0]}, T4: {x[1]}, split: {x[2]}, cr: {x[3]}, far: {x[4]}, pi_pe: {x[5]}')
    opr = x[0]  # overall pressure raio
    T4 = x[1]  # T4 lite onödig men aja. Skulle kunna göra på något smart sätt att T4 alltid sätts till lägsta möjliga
    split = x[2]  # pressure ratio hpc
    cr = x[3]  # compression ratio of the piston engine
    far = x[4]   # fuel air ratio of the piston engine
    pi_pe = x[5]  # piston engine pressure increase or drop
    egr_rate = x[6]  # exhaust gas recirculation rate
    #print(f"before error: {opr}, {T4}, {split}, {cr}, {far}, {pi_pe}, {egr_rate}")

    EXTRA_KEYS = [
    "thrust", "bpr", "bore", "bpr piston", "m0", "p_in_piston",
    "T_in_piston", "T_out_piston", "T35", "P max (bar)",
    "T max", "T_max_twozone", "piston_shaft_power",
    "piston_indicated_power", "piston_heatloss",
    "m_nox_pe", "m_nox_burner", "core_power",
    "core_power_per_litre", "cooling_ratio",
    "piston_fuelflow", "burner_fuelflow", "piston_fuelsplit", 
    "egr_massflow", "equ_piston_in", "piston_massflow", "core_massflow",
    "intercooling_power", "egr_cooling_power", "egr_cooler_effectiveness",
    "T25", "T26", "intercooling_T_in", "intercooling_T_out", "egr_cooling_T_out",
    "m_intercooling", "fan_power", "turbine power",
    "error"
    ]  # the names of the extras



    # error is false at the start of each evaluation
    error = False

    lap1 = timer()

    # the variables that are optimsed upon
    cce_input["OPR"] = opr
    cce_input["T4"] = T4
    cce_input["PR"] = split
    cce_input["cr"] = cr
    cce_input["far piston"] = (far / 100) * (44/43)  
    cce_input["pi_pe"] = pi_pe
    cce_input["EGR_rate"] = egr_rate

    # use the OG efficiencies for assessing size effect
    cce_input["eta_p_hpc"] = eta_p_hpc_0
    cce_input["eta_lpt"] = eta_lpt_0


    # run once to get specific power and linear output temperature from piston engine
    cce_input["life_hack"] = "Simulate"
    # bpr 20 will almost certainly work
    cce_input["bpr"] = 20
    output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

    if output_dict["error"]:
        # if something does not work in the simulation
        error = True

    else:

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
            # if no BPR is found that matches thrust (OR any other error)
            error = True

        else:

            # Use the found bore and BPR for the final evaluation
            cce_input["bpr"] = dict["bpr"][0]
            cce_input["bore"] = dict["bore_match"]


            # final simulation with known bore and BPR to get all info and especially NOX
            cce_input["life_hack"] = "Simulate_final"
            #print("Final simulation")
            try:
                output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)
            except RuntimeError:
                error = True
            if output_dict["error"]:
                error = True



    lap2 = timer()
    print(f"Simulation time for 1 point: {lap2 - lap1} seconds")

    # if point did not converge
    if error:
        specific_nox = 999
        eta_th = 0.0

        # all extras just become zero
        extras = {key: 0 for key in EXTRA_KEYS}

    else:
        # if point converged
        
        #objective
        eta_th = output_dict["thermal efficiency"]
        #specific nox already in mg/Ns
        specific_nox = output_dict["thrust specific nox"]*1e6

        #extras
        bore = output_dict["bore"]
        bpr = dict["bpr"][0]
        thrust = output_dict["thrust"]
        m0 = output_dict["mass flow"]
        T_out_piston = output_dict["T34"]
        T_in_piston = output_dict["T31"]
        p_in_piston = output_dict["p31"]
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
        core_power_per_litre = (core_power/displacement)*1e-6 #kW/litre
        piston_fuelflow = output_dict["piston fuelflow"]
        burner_fuelflow = output_dict["burner fuelflow"] 
        piston_fuelsplit = piston_fuelflow / (piston_fuelflow + burner_fuelflow)

        egr_massflow = output_dict["EGR mass flow"]
        egr_cooler_effectiveness = output_dict["EGR cooler effectiveness"]
        equ_piston_in = output_dict["equ32"]
        piston_massflow = output_dict["piston mass flow"]
        core_massflow = output_dict["core mass flow"]
        intercooling_power = output_dict["intercooler power"]
        egr_cooling_power = output_dict["EGR cooling power"]
        T25 = output_dict["T25"]
        T26 = output_dict["T26"]
        intercooling_T_in = output_dict["intercooler T_in"]
        intercooling_T_out = output_dict["intercooler T_out"]
        egr_T_out = output_dict["EGR cooler T_out"]
        m_intercooling = output_dict["EGR cooler massflow"]
        fan_power = output_dict["fan power"]
        turbine_power = output_dict["turbine power"]





        extras = {
        "thrust": thrust,
        "bpr": bpr,
        "bore": bore,
        "bpr piston": bpr_piston,
        "m0": m0,
        "p_in_piston": p_in_piston * 1e-5,
        "T_in_piston": T_in_piston,
        "T_out_piston": T_out_piston,
        "T35": T35,
        "P max (bar)": pmax * 1e-5,
        "T max": T_max,
        "T_max_twozone":T_max_twozone,
        "piston_shaft_power": piston_shaft_power,
        "piston_indicated_power":piston_indicated_power,
        "piston_heatloss":piston_heatloss,
        "m_nox_pe":m_nox_pe,
        "m_nox_burner":m_nox_burner,
        "core_power":core_power,
        "core_power_per_litre":core_power_per_litre,
        "cooling_ratio":cooling_ratio,
        "piston_fuelflow":piston_fuelflow,
        "burner_fuelflow":burner_fuelflow,
        "piston_fuelsplit":piston_fuelsplit,
        "egr_massflow": egr_massflow,
        "equ_piston_in": equ_piston_in,
        "piston_massflow": piston_massflow,
        "core_massflow": core_massflow,
        "intercooling_power": intercooling_power,
        "egr_cooling_power": egr_cooling_power,
        "egr_cooler_effectiveness": egr_cooler_effectiveness,
        "T25": T25,
        "T26":  T26,
        "intercooling_T_in": intercooling_T_in, 
        "intercooling_T_out": intercooling_T_out,
        "egr_cooling_T_out": egr_T_out,
        "m_intercooling": m_intercooling,
        "fan_power": fan_power,
        "turbine power": turbine_power,
        "error": error
        }

        print(f'opr: {x[0]}, T4: {x[1]}, split: {x[2]}, cr: {x[3]}, far: {x[4]}, pi_pe: {x[5]}, EGR_rate: {x[6]}')
        print(f"Point converged and: thermal efficiency {eta_th*100} % and specific nox: {specific_nox} mg/Ns")
        print(f"Constraints: Pmax {pmax * 1e-5} bar, Tout piston {T_out_piston} K, bore: {bore*1000} mm, bpr around piston: {bpr_piston}")
        print(f"Core power per litre: {core_power_per_litre} kW/l and percentage fuel in piston: {piston_fuelsplit*100} %")


    # Return both objectives (for NSGA-II) and extras (for logging/analysis)
    objectives = np.array([-eta_th, specific_nox])

    
    return objectives, extras



# Defining the problem

#TODO: could remove intercooling from variables
#TODO: add intercooling power, EGR Power and different temepratures in the data that is saved

class MyEngineProblem(Problem):
    def __init__(self):
        super().__init__(n_var=7,  #OPR, T4, split, cr, far, pi_pe, EGR
                            n_obj=2,
                            n_constr=4,  # Tout piston, pmax, circumventing flow, bore?
                            xl=np.array([10, 900, 0.0, 4, 2, 1.1, 0.0]), #lower limit on variables
                            xu=np.array([30, 1400, 0.5, 15, 5, 1.6, 0.5])) #upp limit on the variables

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        G = []
        self.extras = []

        for ind in x:
            obj, extra = evaluate_cce(ind)

            # maybe add constraints regarding fuel air ratio and EGR or T4 or something
            constraints = [
                extra["T_out_piston"] - 1250,  # T_out_piston ≤ 1250
                extra["P max (bar)"] - 150,  # P_max ≤ 50 bar
                extra["bore"] - 0.2, #bore < 200mm
                -extra["bpr piston"], #flow in piston must be equal or smaller than core flow
            ]

            F.append(obj)
            G.append(constraints)
            self.extras.append(extra)
            # Store all evaluated points
            evaluation_record = {
                "opr": ind[0],
                "T4": ind[1],
                "split": ind[2],
                "cr": ind[3],
                "far": ind[4],
                "p_ratio": ind[5],
                "egr_rate": ind[6],
                "eta_th": -obj[0],
                "specific_nox": obj[1],
                "thrust": extra["thrust"],
                "bpr": extra["bpr"],
                "bore": extra["bore"],
                "bpr piston (bar)": extra["bpr piston"],
                "m0": extra["m0"],
                "p_in_piston": extra["p_in_piston"],
                "T_in_piston": extra["T_in_piston"],
                "T_out_piston": extra["T_out_piston"],
                "T35": extra["T35"],
                "P max (bar)": extra["P max (bar)"],
                "T max": extra["T max"],
                "T_max_twozone": extra["T_max_twozone"],
                "piston_shaft_power": extra["piston_shaft_power"],
                "piston_indicated_power": extra["piston_indicated_power"],
                "piston_heatloss": extra["piston_heatloss"],
                "m_nox_pe": extra["m_nox_pe"],
                "m_nox_burner": extra["m_nox_burner"],
                "core_power": extra["core_power"],
                "core_power_per_litre": extra["core_power_per_litre"],
                "cooling_ratio": extra["cooling_ratio"],
                "piston_fuelflow": extra["piston_fuelflow"],
                "burner_fuelflow": extra["burner_fuelflow"],
                "piston_fuelsplit": extra["piston_fuelsplit"],
                "egr massflow": extra["egr_massflow"],
                "equ_piston_in": extra["equ_piston_in"],
                "piston massflow": extra["piston_massflow"],
                "core massflow": extra["core_massflow"],
                "intercooling power": extra["intercooling_power"],
                "egr_cooling_power": extra["egr_cooling_power"],
                "egr_cooler_effectiveness": extra["egr_cooler_effectiveness"],
                "T25": extra["T25"],
                "T26": extra["T26"],
                "intercooling_T_in": extra["intercooling_T_in"],
                "intercooling_T_out": extra["intercooling_T_out"],
                "egr_cooling_T_out": extra["egr_cooling_T_out"],
                "m intercooling": extra["m_intercooling"],
                "fan power": extra["fan_power"],
                "turbine power": extra["turbine power"],
                "error": extra["error"],
                "constraint_violation": max(0, extra["T_out_piston"] - 1250) + max(0, extra["P max (bar)"] - 150) +  max(0, extra["bore"] - 0.2) +  max(0, -extra["bpr piston"]),
                "is_feasible": (extra["T_out_piston"] <= 1250 and
                extra["P max (bar)"] <= 150 and extra["bore"] <= 0.2 and extra["bpr piston"] > 0)
            }
            all_evaluations.append(evaluation_record)

        out["F"] = np.array(F)
        out["G"] = np.array(G)  # Constraint violations: ≤ 0 is feasible



class OptimisationCallback(Callback):
    def __init__(self, ref_point, seed, gen_history=None, hv_history=None):
        super().__init__()
        self.ref_point = ref_point
        self.seed = seed  # store seed
        self.hv_history = hv_history or []   # load existing or start fresh
        self.gen_history = gen_history or []  # load existing or start fresh
        # Fixed offset from previous runs, not updated during this run
        self.gen_offset = gen_history[-1] if gen_history else 0

    def notify(self, algorithm):
        gen = algorithm.n_gen + self.gen_offset  # offset is fixed, no bug
        #output_dir = f"optimisation_data/seed_{self.seed}"
        output_dir = f"."

        # --- Save CSVs every generation ---
        all_df = pd.DataFrame(all_evaluations)
        all_df.to_csv(f"{output_dir}/all_evaluations.csv", index=False)

        if algorithm.opt is not None:
            pareto_df = pd.DataFrame(
                np.hstack([algorithm.opt.get("X"), algorithm.opt.get("F")]),
                columns=['opr', 'T4', 'split', 'cr', 'far', 'pi_pe', 'egr_rate', 'eta_th', 'specific_nox']
            )
            pareto_df['eta_th'] = -pareto_df['eta_th']  # ← flip back to positive
            pareto_df.to_csv(f"{output_dir}/pareto_solutions.csv", index=False)

        # --- Hypervolume ---
        hv_indicator = HV(ref_point=self.ref_point)
        hv = hv_indicator(algorithm.opt.get("F"))
        self.hv_history.append(hv)
        self.gen_history.append(gen)

        # --- Save hypervolume history every generation ---
        hv_df = pd.DataFrame({
            'generation': self.gen_history,
            'hypervolume': self.hv_history,
        })
        hv_df.to_csv(f"{output_dir}/hypervolume.csv", index=False)

        # --- Print monitoring ---
        pareto_size = len(algorithm.opt)
        feasible_total = sum(all_df['is_feasible']) if 'is_feasible' in all_df.columns else '?'
        print(f"Gen {gen:3d} | HV: {hv:.4f} | Pareto size: {pareto_size:3d} | "
              f"Total evals: {len(all_evaluations)} | Feasible: {feasible_total} ")



resume_optimisation = False
# number of generations first run
n_gen = 50

# number of new generations to run
new_gens = 10

pop_size = 50


# save in seed folder
#output_dir = f"optimisation_data/seed_{seed}"
output_dir = f"."
#os.makedirs(output_dir, exist_ok=True)  # add this line

# load previous evlautions
if resume_optimisation and os.path.exists(f"{output_dir}/all_evaluations.csv"):
    existing_df = pd.read_csv(f"{output_dir}/all_evaluations.csv")
    all_evaluations.extend(existing_df.to_dict('records'))


# --- Load existing hypervolume history if it exists ---
hv_csv_path = f"{output_dir}/hypervolume.csv"

if resume_optimisation:
    hv_df_existing = pd.read_csv(hv_csv_path)
    existing_gen = hv_df_existing['generation'].tolist()
    existing_hv = hv_df_existing['hypervolume'].tolist()
else:
    existing_gen = []
    existing_hv = []




# --- Set reference point ---
# Should be worse than any solution you expect:
# eta_th is negative (we minimise -eta_th), so ref should be close to 0 (= 0% efficiency)
# specific_nox ref should be above your worst expected value

# all points below -0.4 eff (remember its negative) and 1.5 nox
ref_point = np.array([-0.4, 1.5])  # adjust nox ref based on your data range

callback = OptimisationCallback(
    ref_point=ref_point,
    seed=seed,
    gen_history=existing_gen,
    hv_history=existing_hv
)

# kör 10 gen 50 pop (sen 100 pop och 50 gen) (testa att köra den på andra datorn kanske) (nej kör 100 pop och 25 gen)
problem = MyEngineProblem()
algorithm = NSGA2(pop_size=pop_size)



if resume_optimisation == False:
    # --- Run optimisation ---
    t_start = timer()

    res = minimize(problem,
                algorithm,
                ('n_gen', n_gen),
                seed=seed,
                verbose=True,
                callback=callback)

    t_end = timer()
    t_total = t_end - t_start


# continue opitimisation
else:
    # --- To resume, load and inject ---
    pop_X = np.loadtxt(f"{output_dir}/last_population_X.csv", delimiter=",")
    pop_F = np.loadtxt(f"{output_dir}/last_population_F.csv", delimiter=",")
    gen_done = int(np.loadtxt(f"{output_dir}/last_generation.csv", delimiter=","))

    # Reconstruct population
    pop = Population.new("X", pop_X, "F", pop_F)

    algorithm = NSGA2(pop_size=pop_size)
    algorithm.initialization.sampling = pop


    # --- Run optimisation ---
    t_start = timer()

    res = minimize(problem,
                            algorithm,
                            ('n_gen', new_gens),
                            seed=seed,
                            verbose=True,
                            callback=callback)
    

    t_end = timer()
    t_total = t_end - t_start




# --- After run, save the final population ---
pop_X = res.pop.get("X")
pop_F = res.pop.get("F")
np.savetxt(f"{output_dir}/last_population_X.csv", pop_X, delimiter=",")
np.savetxt(f"{output_dir}/last_population_F.csv", pop_F, delimiter=",")

# Save total generations done (not just this run's count)
gen_offset = gen_done if resume_optimisation else 0
np.savetxt(f"{output_dir}/last_generation.csv", [res.algorithm.n_gen + gen_offset], delimiter=",")




    #
# Create DataFrame with all evaluated points
all_df = pd.DataFrame(all_evaluations)

# Save all evaluations
all_df.to_csv(f"{output_dir}/all_evaluations.csv", index=False)

# Create DataFrame for Pareto-optimal solutions only
# res.F has the objective values for Pareto solutions
# res.X has the corresponding design variables
pareto_df = pd.DataFrame(
    np.hstack([res.X, res.F]),
    columns=['opr','T4','split','cr','far','pi_pe','egr_rate','eta_th','specific_nox']
)
pareto_df['eta_th'] = -pareto_df['eta_th']  # ← flip back to positive


pareto_df.to_csv(f"{output_dir}/pareto_solutions.csv", index=False)


# save the hypervolume
hv_df = pd.DataFrame({
    'generation': callback.gen_history,
    'hypervolume': callback.hv_history,
})

# save hypervolume and solutons
hv_df.to_csv(f"{output_dir}/hypervolume.csv", index=False)

# --- Print summary ---
hours = int(t_total // 3600)
minutes = int((t_total % 3600) // 60)
seconds = t_total % 60

# Print summary
print(f"\nOptimization Summary:")
print(f"Total run time: {hours}h {minutes}m {seconds:.1f}s")
print(f"Total evaluations: {len(all_df)}")
print(f"Feasible solutions: {sum(all_df['is_feasible'])}")
print(f"Pareto-optimal solutions: {len(pareto_df)}")




