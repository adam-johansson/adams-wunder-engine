def dmvdphi(phi, phi_open, phi_close, n_valve, cd, d, p, T, gamma, R, p3, t3, type):
    if phi > phi_open and phi < phi_close:
        if type == "out":
            p1 = p3
            p2 = p
            t2 = T
        elif type == "in":
            p1 = p
            p2 = p3
            t2 = t3
        else:
            print("Unknown valve")
            return
        l_valve = 1  # valve openning function
        d_valve = 0.75 * (np.sqrt(2) - 1) * d
        A_valve = n_valve * cd * np.pi * d_valve**2 * l_valve
        dtdphi = s / (np.pi * v_mean)
        pi = min(p2 / p1, (2 / (gamma + 1)) ** (gamma / (gamma - 1)))
        FF = np.sqrt(
            gamma / (gamma - 1) * (pi ** (2 / gamma) - pi ** ((gamma + 1) / gamma))
        )
        return A_valve * np.sqrt(2 / (R * t2)) * p2 * FF * dtdphi
    else:
        return 0.0
