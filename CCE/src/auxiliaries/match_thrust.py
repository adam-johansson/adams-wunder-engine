import numpy as np

from CCE.src import cce_propulsion_system, cce_propulsion_system_specific
from timeit import default_timer as timer

from scipy.optimize import (
    fsolve, brentq
)

# THIS IS THE MOST UP TO DATE VERSION


def run_cce_fpr(input, data_piston, flags, meta_model):

    Fs_goal = input["Fs_req"]  # specific thrust
    x0 = np.array([1.2])
    limits = [(1.15, 1.7)]

    outputs = 14

    check = []

    def find_fpr(fpr):
        if fpr > 2.0 or fpr < 1.0:
            return 1e6
        input["fpr_outer"] = fpr[0]  # fpr
        # print(fpr[0])
        start = timer()
        cce_output_dict = cce_propulsion_system.run_cce(input, data_piston, flags, meta_model)
        end = timer()
        error = cce_output_dict["error"]
        sfc = cce_output_dict["sfc"]
        # print(f'one run cce: {end - start}')
        if error:
            # print(sfc, fpr)
            # print('Not working power plant.')
            check.append(sfc)
            if len(check) > 4:
                #print(f'Error funkar: {check}')
                raise ValueError
            return sfc
        # cost = np.abs(thrust / m0 - Fs_goal)
        thrust = cce_output_dict["thrust"]
        m0 = cce_output_dict["mass flow"]
        cost = thrust / m0 - Fs_goal
        #print(f"Residual between specific thrust and goal thrust: {cost}. Outer FPR: {fpr}")
        return cost

    try:
        # start = timer()
        # fsolve is slightly faster but doesn't converge as well

        opt_fpr, infodict, ier, mesg = fsolve(
            find_fpr, x0, full_output=True, epsfcn=1e-9
        )
        if ier == 1:
            sol = infodict["fvec"]
            # print(f'Fsolve worked. {sol}, {opt_fpr[0]}')

        elif ier != 1:
            #print(ier, mesg)
            error = True
            #cost = 15e-6 + np.abs(infodict["fvec"]) * 1e3
            cost = 999
            listofzeros = [0] * outputs
            listofzeros[-1] = error
            listofzeros[0] = cost
            return listofzeros

        # end = timer()
        # print(f'Time: {end-start}')

    except ValueError as e:
        error = True
        #print(e)
        cost = 999
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros



    # FINAL RUN
    #flags.append("print_output")
    #print(f"Matching outer FPR is: {opt_fpr[0]}")
    input["fpr_outer"] = opt_fpr[0]

    cce_output_dict = cce_propulsion_system.run_cce(input, data_piston, flags, meta_model)

    thrust_final = cce_output_dict["thrust"]
    m0 = cce_output_dict["mass flow"]


    output_dict = cce_output_dict
    output_dict["fpr"] = opt_fpr[0]


    if np.abs(thrust_final / m0 - Fs_goal) > 1e-3:
        # if thrust didnt match raise error
        error = True
        #print('andra error funkar')
        cost = sfc_final + np.abs(thrust_final / m0 - Fs_goal) * 1e3
        cost = 999
        output_dict["sfc"] = cost
        output_dict["error"] = error

        return output_dict

    return output_dict


def run_cce_bpr(input, data_piston, meta_model):
    Fs_goal = input["Fs_req"]  # specific thrust

    flags = ["single", "cce"]

    error = False
    piston_error = False  # Flag to track piston errors

    def find_bpr(x):
        nonlocal piston_error  # Allow modification of outer scope variable

        input["bpr"] = x

        output_dict = cce_propulsion_system_specific.run_cce(input, data_piston, flags, meta_model)

        if output_dict["error"]:
            if output_dict["error_type"] == "LPT" or output_dict["error_type"] == "Hot nozzle":
                # Too high BPR means LPT does not work
                # Return very low thrust to guide brentq to lower BPR
                Fs = 0.0
            elif output_dict["error_type"] == "piston" or output_dict["error_type"] == "T4":
                # Set flag and raise exception to break out of brentq
                piston_error = True
                raise ValueError("Piston error encountered")
            else:
                # Other error - return low thrust
                Fs = 0.0
        else:
            Fs = output_dict["specific thrust"]

        residual = np.array([Fs - Fs_goal])
        return residual

    try:
        bpr = brentq(find_bpr, 5, 25)
    except ValueError as e:
        if piston_error:
            # Piston error occurred, set error flag and return
            error = True
            bpr = None
        else:
            # Some other ValueError, re-raise it
            raise e
    except ValueError:
        # brentq failed to converge (no piston error)
        error = True
        bpr = None

    output = {
        "bpr": bpr,
        "error": error
    }

    return output  # Return the full output dict, not just bpr


