import numpy as np

from CCE.src import cce_propulsion_system
from timeit import default_timer as timer

from scipy.optimize import minimize, minimize_scalar, brentq, differential_evolution, fsolve

# THIS IS THE MOST UP TO DATE VERSION


def run_cce_fpr(data, data_piston, flags, meta_model):

    Fs_goal = data[5]  # specific thrust
    x0 = np.array([1.2])
    limits = [(1.15, 1.7)]

    check = []

    def find_fpr(fpr):
        if fpr > 2.0 or fpr < 1.0:
            return 1e6
        data[4] = fpr[0]  # fpr
        #print(fpr[0])
        start = timer()
        sfc, v_ratio, thrust, m0, error, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston\
            = cce_propulsion_system.run_cce(data, data_piston, flags, meta_model)
        end = timer()
        #print(f'one run cce: {end - start}')
        if error:
            #print(sfc, fpr)
            #print('Not working power plant.')
            check.append(sfc)
            if len(check) > 4:
                #print(f'Error funkar: {check}')
                raise ValueError
            return sfc
        #cost = np.abs(thrust / m0 - Fs_goal)
        cost = thrust / m0 - Fs_goal
        #print(cost, fpr)
        return cost

    try:
        #start = timer()
        # fsolve is slightly faster but doesn't converge as well
        #opt_fpr = minimize(find_fpr, x0, bounds=limits, method='Nelder-Mead', options={'disp': False, 'fatol': 1e-5})
        opt_fpr, infodict, ier, mesg = fsolve(find_fpr, x0, full_output=True, epsfcn=1e-9)
        if ier == 1:
            sol = infodict['fvec']
            #print(f'Fsolve worked. {sol}, {opt_fpr[0]}')

        elif ier != 1:
            #print(ier, mesg)
            cost = 5e-6 + np.abs(infodict['fvec']) * 1e-5
            return cost, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,


        #end = timer()
        #print(f'Time: {end-start}')

    except ValueError as e:
        error = True
        #print(e)
        return 999, 0, 0, 1, error, 0, 0, 0, 0, 0, 0, 0

    #opt_fpr = minimize(find_fpr, x0, bounds=limits, method='Nelder-Mead', options={'disp': True, 'fatol': 1e-0})
    #opt_fpr = minimize_scalar(find_fpr, x0, bounds=limits)
    #opt_fpr = differential_evolution(find_fpr, bounds=limits)

    data[4] = opt_fpr[0]

    sfc_final, v_ratio_final, thrust_final, m0, error, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston\
        = cce_propulsion_system.run_cce(data, data_piston, flags, meta_model)

    if np.abs(thrust_final / m0 - Fs_goal) > 0.1:
        # if thrust didnt match raise error
        error = True
        print('andra error funkar')
        cost = sfc_final + np.abs(thrust_final / m0 - Fs_goal)
        return cost, 0, thrust_final, m0, error, opt_fpr[0], p_max, T_max, T_in_piston, T_out_piston,\
        TET, far_piston

    return sfc_final, v_ratio_final, thrust_final, m0, error, opt_fpr[0], p_max, T_max, T_in_piston, T_out_piston,\
        TET, far_piston
