import numpy as np


def dmvdphi(phi, phi_open, phi_close, n_valve, lv_max, cd, d, p, T, gamma, R, p3, T3, gamma3, R3, type, phi_cycle):
    if phi_open < phi < phi_close:
        backflow = False
        if type == 'out':
            if p / p3 > 1:
                p1 = p
                p2 = p3
                T1 = T
                R1 = R
                gamma1 = gamma
            else:
                p1 = p3
                p2 = p
                T1 = T3
                R1 = R3
                gamma1 = gamma3
                backflow = True
        elif type == 'in':
            if p3 / p > 1:
                p1 = p3
                p2 = p
                T1 = T3
                R1 = R3
                gamma1 = gamma3
            else:
                p1 = p
                p2 = p3
                T1 = T
                gamma1 = gamma
                R1 = R
                backflow = True
        else:
            print('Unknown valve')
            return

        phi_duration = phi_close - phi_open
        phi_rel = phi - phi_open
        f = phi_rel / phi_duration

        if f < 0.25:
            lv = lv_max * 0.5 * (1 - np.cos(4 * np.pi * f))
        elif f < 0.75:
            lv = lv_max
        else:
            lv = lv_max * 0.5 * (1 - np.cos(4 * np.pi * (1 - f)))

        if gamma1 < 1.1:
            print(gamma1, T1, p1, R1, backflow, type)
        PI = max(p2 / p1, (2 / (gamma1 + 1)) ** (gamma1 / (gamma1 - 1)))
        FF = np.sqrt((gamma1 / (gamma1 - 1)) * (PI ** (2 / gamma1) - PI ** ((gamma1 + 1) / gamma1)))
        dvalve = 0.75 * (np.sqrt(2) - 1) * d

        A = n_valve * cd * np.pi * dvalve * lv
        if backflow:
            return -A * np.sqrt(2 / (R1 * T1)) * p1 * FF
        else:
            return A * np.sqrt(2 / (R1 * T1)) * p1 * FF

    return 0.0
