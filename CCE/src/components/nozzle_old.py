import numpy as np


def nozzle_old(p1, T1, pa, eta, cp, gamma, ms):
    R = cp * (gamma - 1) / gamma

    pc = p1 * (1 - (1 / eta) * ((gamma - 1) / (gamma + 1))) ** (gamma / (gamma - 1))

    if pa < pc:
        choked = True
        p2 = pc
        T2 = T1 * 2 / (gamma + 1)
        c2 = np.sqrt(gamma * R * T2)
        rho2 = p2 / (R * T2)
        As = ms / (rho2 * c2)
    else:
        choked = False
        p2 = pa
        T2 = T1 - eta * T1 * (1 - (1 / (p1 / p2)) ** ((gamma - 1) / gamma))
        print(f"Nozzle is not choked and total pressure ratio is: {p1/pa}")
        c2 = np.sqrt(2 * cp * (T1 - T2))
        As = 0

    Fs = c2 * ms + (p2 - pa) * As

    c_ideal = np.sqrt(2 * cp * T1 * (1 - (1 / (p1 / pa)) ** ((gamma - 1) / gamma)))

    return choked, Fs, p1 / pa, c2, c_ideal
