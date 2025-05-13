import numpy as np

from CCE.src import cce_propulsion_system
from timeit import default_timer as timer

from scipy.optimize import (
    minimize,
    minimize_scalar,
    brentq,
    differential_evolution,
    fsolve,
)


def run_cce_bpr_fpr(data, flags):

    Fs_goal = data[5]  # specific thrust
    v_ratio_goal = 0.86  # ratio of ideal jet velocities
    x0 = np.array([26, 1.27])
    limits = ([20, 30], [1.20, 1.35])

    def find_bpr_fpr(x):
        data[2] = x[0]  # bpr
        data[4] = x[1]  # fpr
        # print(f'BPR: {data[2]}, FPR: {data[4]}')
        sfc, v_ratio, thrust, m0, error = cce_propulsion_system.run_cce(data, flags)
        if error:
            print("Not working power plant.")
            return 99999
        # print(f'Fs: {thrust / m0}, v_ratio: {v_ratio}, sfc: {sfc * 1e6}')
        cost = np.sqrt(
            (thrust / m0 - Fs_goal) ** 2 + 1e3 * (v_ratio - v_ratio_goal) ** 2
        )
        return cost

    # try:
    #    opt = least_squares(find_bpr_fpr, x0, bounds=limits)
    # except ValueError:
    #    print('problem with matching bpr and fpr to specific thrust and optimal velocity ratio')
    #    error = True
    #    return 999, 0, 0, 1, error

    opt = minimize(
        find_bpr_fpr,
        x0,
        bounds=limits,
        method="Nelder-Mead",
        options={"disp": True, "fatol": 1e-0},
    )

    data[2] = opt.x[0]
    data[4] = opt.x[1]

    sfc_final, v_ratio_final, thrust_final, m0, error = cce_propulsion_system.run_cce(
        data, flags
    )

    return sfc_final, v_ratio_final, thrust_final, m0, error
