import numpy as np

from CCE.src import cce_propulsion_system
from CCE.src import auxiliaries

from scipy.optimize import least_squares, brentq, fsolve, differential_evolution


def run_cce_v_ratio(data, flags):

    # runs the cce cycle, but ensures that the ideal jet velocity ratio is optimal

    # INSERT FUNCTION THAT CALCULATES OPTIMUM V_RATIO

    def find_fpr(fpr):
        data[4] = fpr  # fpr
        sfc, v_ratio, thrust, m0, error = cce_propulsion_system.run_cce(data, flags)
        if error:
            print("Not working power plant.")
            return 99999
        return v_ratio - 0.78

    try:
        fpr_opt = brentq(find_fpr, 1.1, 2.0)
    except ValueError:
        print("problem with matching fpr for optimum vel ratio")
        error = True
        return 999, 0, 0, 0, error

    data[5] = fpr_opt

    sfc_final, v_ratio_final, thrust_final, m0, error = cce_propulsion_system.run_cce(
        data, flags
    )

    return sfc_final, v_ratio_final, thrust_final, m0, error
