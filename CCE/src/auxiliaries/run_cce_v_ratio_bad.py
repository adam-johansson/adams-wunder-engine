import numpy as np

from CCE.src import cce_propulsion_system
from CCE.src import auxiliaries

from scipy.optimize import least_squares, brentq


def run_cce_v_ratio(data, flags):

    # runs the cce cycle, but ensures that the ideal jet velocity ratio is optimal

    # finding fpr that gives required thrust
    def find_fpr(fpr):
        data_temp = data
        data_temp[5] = fpr
        sfc, v_ratio, thrust = cce_propulsion_system.run_cce(data_temp, flags)
        if np.isnan(sfc):
            return 99999
        F_req = data[0]
        print(thrust, fpr)
        return thrust - F_req
        #return v_ratio - 0.78

    def find_fpr_vratio(fpr):
        data_temp = data
        data_temp[5] = fpr
        sfc, v_ratio, thrust = cce_propulsion_system.run_cce(data_temp, flags)
        if np.isnan(sfc):
            return 99999
        print(v_ratio, fpr)
        return v_ratio - 0.78

    def find_m(m):
        data_temp = data
        data_temp[29] = m
        sfc, v_ratio, thrust = cce_propulsion_system.run_cce(data_temp, flags)
        if np.isnan(sfc):
            return 99999
        F_req = data[0]
        print(thrust, m)
        return thrust - F_req

    def find_bpr(bpr):
        print(data[5])
        data_temp = data
        data_temp[2] = bpr
        sfc, v_ratio, thrust = cce_propulsion_system.run_cce(data_temp, flags)
        if np.isnan(sfc):
            return 99999

        print(v_ratio, bpr)
        return v_ratio - 0.78

    # can make this faster by having a better guess. can calculate fpr beforehand
    x0 = np.array([1.4])
    #opt_fpr = least_squares(find_fpr, x0[0], bounds=([1.1, 2.0])).x[0]
    fpr_opt = brentq(find_fpr_vratio, 1.1, 2.0)
    #opt_fpr = auxiliaries.fmatch(find_fpr, 1.3, 1.8, tol=1e-3)
    data[5] = fpr_opt

    #bpr_opt = brentq(find_bpr, 15, 30)
    #data[2] = bpr_opt

    m_opt = brentq(find_m, 100, 300)
    data[29] = m_opt

    sfc_final, v_ratio_final, thrust_final = cce_propulsion_system.run_cce(data, flags)

    print(v_ratio_final, thrust_final*1e-3, m_opt, fpr_opt)
    return sfc_final
