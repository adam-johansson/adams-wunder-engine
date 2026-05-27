import numpy as np

from CCE.src import cce_propulsion_system, cce_propulsion_system_specific
from timeit import default_timer as timer

from CCE.src.misc import thrust_requirement_function

from mpmath.libmp import fzero
from scipy.optimize import (
    fsolve, brentq
)

def run_cce_bpr(input, data_piston, meta_model):
    # baseline thrust goal
    F_goal = input["Fn"]  # net thrust for baseline engine



    # trade factor interpolator
    interpolator = input["trade_factor_interpolator"]

    if meta_model == "placeholder":
        flags = ["life_hack", "cce"]
    else:
        flags = ["single", "cce"]

    error = False
    piston_error = False  # Flag to track piston errors
    error_type = False

    # Store the last bore match
    last_outputs = {}
    def find_bpr(x):
        nonlocal piston_error, error, error_type, F_goal  # Allow modification of outer scope variable
        #print(f"Thrust required: {F_goal*1e-3} kN")

        input["bpr"] = x[0]
        input["Fn"] = F_goal

        output_dict = cce_propulsion_system_specific.run_cce(input, data_piston, flags, meta_model)


        if output_dict["error"]:
            #print(output_dict["error_type"])
            if output_dict["error_type"] == "LPT" or output_dict["error_type"] == "Hot nozzle":
                #print(f"fel2")
                # Too high BPR means LPT does not work
                # Return very low thrust to guide brentq to lower BPR
                F = 0.0
                bore = 999
                error_type = output_dict["error_type"]
                error = True

            elif output_dict["error_type"] == "PISTON" or output_dict["error_type"] == "T4" or output_dict["error_type"] == "second burner temp":
                # Too low BPR means piston needs to do too much work and it will fail (need too large bore)
                # larger piston also means higher T35 wich could exceed T4
                # return very high thrust to guide brentq to higher BPR
                #print("hej")
                F = 9999999
                bore = 999
                error_type = output_dict["error_type"]
                error = True
                
                
                #raise ValueError("Piston error encountered")
            else:
                # Other error - return low thrust
                #print(f"fel")
                F = 0.0
                bore = 999

            # this is used since the real F_goal changes 
            F_goal_dummy = 27000
            residual = np.array([F - F_goal_dummy])

            return residual

        else:
            F = output_dict["thrust"]
            bore = output_dict["bore"]
            sfc = output_dict["sfc"]
            weight = output_dict["weight"]
            error = False
            error_type = False

            # calculate new thrust requirement based on sfc and engine weight
            F_goal = thrust_requirement_function(sfc,weight,interpolator)

            # denna var med även vid error förut så det kanske kan bli problem
            last_outputs.update({
                'bore_match': bore,
            })

            residual = np.array([F - F_goal])
            #print(f"Thrust residual {residual} bpr {x} bore: {bore}")
            return residual


    try:
        #bpr = brentq(find_bpr, 12, 40)
        bpr = fsolve(find_bpr, x0=15)
    except ValueError:
        error = True
        bpr = None
        last_outputs['bore_match'] = None
        error_type = "Can't match thrust"

    output = {
        "bpr": bpr,
        "bore_match": last_outputs['bore_match'],
        "error": error,
        "error_type": error_type
    }

    return output  # Return the full output dict, not just bpr


