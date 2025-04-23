from CCE.src import thermo_outdated
import numpy as np


def fpr_opt(Mach, bpr, Fs_req, Ta, eta_lpt, eta_fan, cfg_bypass, cd_nozzle, dp_bypass):

    R_uni = 8.3144626  # J mol^-1 K^-1

    cp, h, s, M = thermo_outdated.properties(Ta, equ=0)
    R = R_uni / M
    cv = cp - R
    gamma = cp / cv
    # this could probably include pressure loss from heat exchanger in the future. also right now it is the
    # polytropic efficiency and not isentropic
    eta_nb = cfg_bypass * cd_nozzle * (1 - dp_bypass)
    eta = eta_lpt * eta_fan * eta_nb * 0.9

    a = (gamma - 1) / (2 + (gamma - 1) * Mach**2)
    b1 = ((1 + bpr) ** 2) / ((bpr + 1 / eta) ** 2)
    b2 = (Fs_req / np.sqrt(gamma * R * Ta) + Mach) ** 2

    b = b1 * b2 - Mach**2

    fpr_term = 1 + a * b
    fpr = fpr_term ** (gamma / (gamma - 1))
    return fpr
