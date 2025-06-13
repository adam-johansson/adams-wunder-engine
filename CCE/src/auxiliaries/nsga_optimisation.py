from CCE.src import cce_propulsion_system
from CCE.src import auxiliaries
from CCE.src import misc

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import pandas as pd
import plotly.express as px
import os
import numpy as np
import matplotlib.pyplot as plt


def nsga_optimisation(input, piston_input, flags, meta_model):
    # Global storage for all evaluated points
    all_evaluations = []

    def global_sfc(x):
        # print(f'bpr: {x[0]}, opr: {x[1]}, split: {x[2]}, pi_pe: {x[3]}, cr: {x[4]}, bore: {x[5]}')
        bpr = x[0]  # bypass ratio
        opr = x[1]
        split = x[2]  # pressure ratio hpc
        cr = x[3]
        bore = x[4]
        pi_pe = x[5]  # piston engine pressure increase or drop

        input["bpr"] = bpr
        # data[3] = T35
        input["opr"] = opr
        input["PR"] = split
        input["pi_pe"] = pi_pe
        input["cr"] = cr
        input["bore"] = bore
        # maybe diameter? or match core flow?

        # print('ny iter')

        # sfc, v_ratio, F, m0, error = cce_propulsion_system.run_cce(data, flags)
        sfc, v_ratio, thrust, m0, fpr, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston, T35, EI_nox, error, \
            = auxiliaries.run_cce_fpr(input, piston_input, flags, meta_model)

        if error:
            EI_nox = 999
            sfc = 999

        # if T_out_piston > 1350:
        #    sfc = 999
        #    EI_nox = 999
        #    print(f"Too hot out of piston:{T_out_piston}")

        # Return both objectives (for NSGA-II) and extras (for logging/analysis)
        objectives = np.array([sfc * 1e6, EI_nox])
        extras = {
            "thrust": thrust,
            "m0": m0,
            "T_in_piston": T_in_piston,
            "T_out_piston": T_out_piston,
            "T35": T35,
            "TET": TET,
            "FAR": far_piston,
            "FPR outer": fpr,
            "P max (bar)": p_max * 1e-5,
            "T max": T_max,
            "error": error
        }

        return objectives, extras

    class MyEngineProblem(Problem):
        def __init__(self):
            super().__init__(n_var=6,
                             n_obj=2,
                             n_constr=2,  # for T out piston
                             xl=np.array([10, 10, 0.1, 6, 0.1, 0.9]),
                             xu=np.array([30, 30, 0.9, 12, 0.2, 1.5]))
            # If pi_pe is included, n_var = 6, and adjust bounds accordingly

        def _evaluate(self, x, out, *args, **kwargs):
            F = []
            G = []
            self.extras = []

            for ind in x:
                obj, extra = global_sfc(ind)

                constraints = [
                    extra["T_out_piston"] - 1350,  # T_out_piston ≤ 1350
                    extra["P max (bar)"] - 250,  # P_max ≤ 50 bar
                ]

                F.append(obj)
                G.append(constraints)
                self.extras.append(extra)

                # Store all evaluated points
                evaluation_record = {
                    "bpr": ind[0],
                    "opr": ind[1],
                    "split": ind[2],
                    "cr": ind[3],
                    "bore": ind[4],
                    "p_ratio": ind[5],
                    "sfc": obj[0],
                    "EI_nox": obj[1],
                    "thrust": extra["thrust"],
                    "m0": extra["m0"],
                    "T_in_piston": extra["T_in_piston"],
                    "T_out_piston": extra["T_out_piston"],
                    "T35": extra["T35"],
                    "TET": extra["TET"],
                    "FAR": extra["FAR"],
                    "FPR outer": extra["FPR outer"],
                    "P max (bar)": extra["P max (bar)"],
                    "T max": extra["T max"],
                    "error": extra["error"],
                    "constraint_violation": (extra["T_out_piston"] - 1350 or
                                             extra["P max (bar)"] - 250),
                    "is_feasible": (extra["T_out_piston"] <= 1350 and
                    extra["P max (bar)"] <= 250)
                }
                all_evaluations.append(evaluation_record)

            out["F"] = np.array(F)
            out["G"] = np.array(G)  # Constraint violations: ≤ 0 is feasible

    problem = MyEngineProblem()
    algorithm = NSGA2(pop_size=50)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=True)

    # Create DataFrame with all evaluated points
    all_df = pd.DataFrame(all_evaluations)

    # Save all evaluations
    all_df.to_csv("optimisation_data/all_evaluations.csv", index=False)

    # Create DataFrame for Pareto-optimal solutions only
    pareto_results = []
    for x in res.X:
        obj, extra = global_sfc(x)
        result = {
            "bpr": x[0],
            "opr": x[1],
            "split": x[2],
            "cr": x[3],
            "bore": x[4],
            "p_ratio": x[5],
            "sfc": obj[0],
            "EI_nox": obj[1],
            "thrust": extra["thrust"],
            "m0": extra["m0"],
            "T_in_piston": extra["T_in_piston"],
            "T_out_piston": extra["T_out_piston"],
            "T35": extra["T35"],
            "TET": extra["TET"],
            "FAR": extra["FAR"],
            "FPR outer": extra["FPR outer"],
            "P max (bar)": extra["P max (bar)"],
            "T max": extra["T max"],
            "error": extra["error"]
        }
        pareto_results.append(result)

    pareto_df = pd.DataFrame(pareto_results)
    pareto_df.to_csv("optimisation_data/pareto_solutions.csv", index=False)

    # Print summary
    print(f"\nOptimization Summary:")
    print(f"Total evaluations: {len(all_df)}")
    print(f"Feasible solutions: {sum(all_df['is_feasible'])}")
    print(f"Pareto-optimal solutions: {len(pareto_df)}")

    """# Plotting - All points vs Pareto front
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: All evaluated points
    feasible_mask = all_df['is_feasible']
    infeasible_mask = ~feasible_mask

    if sum(infeasible_mask) > 0:
        scatter1 = ax1.scatter(all_df[infeasible_mask]['sfc'], all_df[infeasible_mask]['EI_nox'],
                               c='red', alpha=0.3, s=20, label='Infeasible', marker='x')

    if sum(feasible_mask) > 0:
        scatter2 = ax1.scatter(all_df[feasible_mask]['sfc'], all_df[feasible_mask]['EI_nox'],
                               c=all_df[feasible_mask]['split'], cmap='viridis',
                               alpha=0.6, s=30, label='Feasible')
        plt.colorbar(scatter2, ax=ax1, label='Compression Split (-)')

    # Highlight Pareto front
    ax1.scatter(pareto_df['sfc'], pareto_df['EI_nox'],
                c='black', s=50, marker='s', label='Pareto Front',
                facecolors='none', edgecolors='black', linewidth=2)

    ax1.set_xlabel("SFC [mg/Ns]")
    ax1.set_ylabel("EI_NOx [g/kg_fuel]")
    ax1.set_title("All Evaluated Points")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Pareto front only (colored by split)
    scatter3 = ax2.scatter(pareto_df['sfc'], pareto_df['EI_nox'],
                           c=pareto_df['split'], cmap='viridis',
                           s=50, edgecolors='k')
    plt.colorbar(scatter3, ax=ax2, label='Compression Split (-)')
    ax2.set_xlabel("SFC [mg/Ns]")
    ax2.set_ylabel("EI_NOx [g/kg_fuel]")
    ax2.set_title("Pareto Front Only")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    """
    return all_df, pareto_df