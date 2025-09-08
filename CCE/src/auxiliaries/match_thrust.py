import numpy as np

from CCE.src import cce_propulsion_system
from timeit import default_timer as timer

from scipy.optimize import (
    fsolve,
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
