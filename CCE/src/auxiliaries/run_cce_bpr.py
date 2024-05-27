import numpy as np

from CCE.src import cce_propulsion_system

from scipy.optimize import brentq


def run_cce_bpr(data, flags):

    # adjusts bpr to match specific thrust for fixed mass flow

    def find_bpr(bpr):
        data[2] = bpr  # fpr
        sfc, v_ratio, thrust, m0, error = cce_propulsion_system.run_cce(data, flags)
        if error:
            print('Not working power plant.')
            return 99999
        return thrust / m0 - 80

    try:
        bpr_opt = brentq(find_bpr, 26, 26.1)
    except ValueError:
        print('problem with matching bpr to specific thrust')
        error = True
        return 999, 0, 0, 0, error

    data[2] = bpr_opt

    sfc_final, v_ratio_final, thrust_final, m0, error = cce_propulsion_system.run_cce(data, flags)

    return sfc_final, v_ratio_final, thrust_final, m0, error
