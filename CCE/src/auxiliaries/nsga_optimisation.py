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


def nsga_optimisation(data, data_piston, flags, meta_model):


    def global_sfc(x):

        # print(f'bpr: {x[0]}, opr: {x[1]}, split: {x[2]}, pi_pe: {x[3]}, cr: {x[4]}, bore: {x[5]}')
        bpr = x[0]  # bypass ratio
        opr = x[1]
        split = x[2]  # pressure ratio hpc
        cr = x[3]
        bore = x[4]
        #pi_pe = x[5]  # piston engine pressure increase or drop


        data[2] = bpr
        # data[3] = T35
        data[27] = opr
        data[28] = split
        #data[27] = pi_pe
        data[26] = cr
        data[29] = bore
        # maybe diameter? or match core flow?

        # print('ny iter')

        # sfc, v_ratio, F, m0, error = cce_propulsion_system.run_cce(data, flags)
        sfc, v_ratio, thrust, m0, fpr, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston, T35, EI_nox, error,\
           = auxiliaries.run_cce_fpr(data, data_piston, flags, meta_model)

        if error:
            EI_nox = 999
            sfc = 999

        #if T_out_piston > 1350:
        #    sfc = 999
        #    EI_nox = 999
        #    print(f"Too hot out of piston:{T_out_piston}")


        # Return both objectives (for NSGA-II) and extras (for logging/analysis)
        objectives = np.array([sfc*1e6, EI_nox])
        extras = {
            "thrust": thrust,
            "m0": m0,
            "T_in_piston": T_in_piston,
            "T_out_piston": T_out_piston,
            "T35": T35,
            "TET": TET,
            "FAR": far_piston,
            "FPR outer": fpr,
            "P max (bar)": p_max*1e-5,
            "T max": T_max,
            "error": error
        }

        return objectives, extras


    class MyEngineProblem(Problem):
        def __init__(self):
            super().__init__(n_var=5,
                             n_obj=2,
                             n_constr=1,  # for T out piston
                             xl=np.array([10, 10, 0.1, 6, 0.1]),
                             xu=np.array([30, 30, 0.9, 12, 0.2]))
            # If pi_pe is included, n_var = 6, and adjust bounds accordingly

        def _evaluate(self, x, out, *args, **kwargs):
            F = []
            G = []
            self.extras = []

            for ind in x:
                obj, extra = global_sfc(ind)
                F.append(obj)
                G.append(extra["T_out_piston"] - 1350)  # Constraint: TET must be ≤ 1350
                self.extras.append(extra)

            out["F"] = np.array(F)
            out["G"] = np.array(G)  # Constraint violations: ≤ 0 is feasible

    problem = MyEngineProblem()

    algorithm = NSGA2(pop_size=300)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 300),
                   seed=1,
                   verbose=True)

    # res.F contains the objectives: shape (n_solutions, 2)
    # res.X contains the decision variables: shape (n_solutions, n_vars)

    for i in range(len(res.F)):
        print(f"Solution {i + 1}:")
        print(f"  Objectives (SFC, EI_NOx): {res.F[i]}")
        print(f"  Design variables (bpr, opr, split, cr, bore): {res.X[i]}")
        print()


    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()

    # Prepare list of dicts with all values
    # Here we run these data points again to get the correct extra values
    results = []
    for x in res.X:
        obj, extra = global_sfc(x)
        result = {
            "bpr": x[0],
            "opr": x[1],
            "split": x[2],
            "cr": x[3],
            "bore": x[4],
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
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("optimisation_data/pareto_solutions.csv", index=False)

    # Unpack results
    X = res.X  # Decision variables
    F = res.F  # Objective values (SFC, EI_NOx)

    # Choose which variable to color by (e.g., bpr is column 0)
    color_by = X[:, 2]  # split

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(F[:, 0], F[:, 1], c=color_by, cmap='viridis', edgecolors='k')
    plt.colorbar(sc, label='Compression split (-)')
    plt.xlabel("SFC [mg/Ns]")
    plt.ylabel("EI_NOx [g/kg_fuel]")
    plt.title("Pareto Front Colored by Compression Split (-)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #df = pd.DataFrame(X, columns=["bpr", "opr", "split", "cr", "bore"])
    #df["sfc"] = F[:, 0]
    #df["EI_nox"] = F[:, 1]

    #fig = px.scatter(df, x="sfc", y="EI_nox", color="bpr", hover_data=df.columns)
    #fig.update_layout(title="Pareto Front Colored by Bypass Ratio")
    #fig.show()

    return
