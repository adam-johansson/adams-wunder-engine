import numpy as np
from numba import jit

def dqfdt_kaiser(Qf, t, t_soc, t_eoc, wa, wm):
    # Wiebe function from Kaiser's thesis
    tc = t_eoc - t_soc
    if t > t_soc and t < t_soc + tc:
        y = (t - t_soc) / tc
        return Qf / (1 - np.exp(-wa)) * wa * (wm + 1) * y ** wm / tc * np.exp(-wa * y ** (wm + 1))
    return 0.0


@jit(nopython=True)
def dqfdt_single(phi, m, phi_sc, phi_cd, Qf):
    # This is the real Wiebe without premixed. Look at Watson paper or Grundlagen or NASA 2T

    # only 99.9% of Qf will be added to the fluid
    if phi > phi_sc:
        # and phi < phi_sc + phi_cd + 0.5:
        y = (phi - phi_sc) / phi_cd
        f1 = 6.908 * (m + 1) * (y ** m) * np.exp(-6.908 * y ** (m + 1))
        return f1 * Qf / phi_cd
    return 0.0


def dmfdphi_double(phi, c1, c2, c3, c4, c5, mf_tot):
    # This is the double Wiebe from NASA
    if phi > c2 and phi < c2 + c3:
        y = (phi - c2) / c3
        f1 = 6.907755 * (c5 + 1) * (y ** c1) * np.exp(-6.907755 * y ** (c1 + 1))
        f2 = 5000 * (c5 + 1) * y ** c5 * (1 - y ** (c5 + 1)) ** 4999
        mdotf = mf_tot * (c4 * f2 + (1 - c4) * f1)
        return mdotf
    return 0.0


def dqfdt_single_vector(phi, m, phi_sc, phi_cd, Qf):
    # This is the real Wiebe without premixed. Look at Watson paper or Grundlagen or NASA 2T

    # only 99.9% of Qf will be added to the fluid

    y = (phi - phi_sc) / phi_cd
    f1 = 6.908 * (m + 1) * (y ** m) * np.exp(-6.908 * y ** (m + 1))

    dqfdphi = f1 * Qf / phi_cd

    idx = dqfdphi < 0

    dqfdphi[idx] = 0.0

    return dqfdphi


@jit(nopython=True)
def dmfdphi_single_mass(phi, m, phi_sc, phi_cd, mf_tot):
    # This is the real Wiebe without premixed. Look at Watson paper or Grundlagen or NASA 2T

    # only 99.9% of Qf will be added to the fluid
    if phi > phi_sc:
        # and phi < phi_sc + phi_cd + 0.5:
        y = (phi - phi_sc) / phi_cd
        f1 = 6.908 * (m + 1) * (y ** m) * np.exp(-6.908 * y ** (m + 1))
        return f1 * mf_tot / phi_cd
    return 0.0


@jit(nopython=True)
def dmfdphi_single_mass_vector(phi, m, phi_sc, phi_cd, mf_tot):
    # This is the real Wiebe without premixed. Look at Watson paper or Grundlagen or NASA 2T

    # only 99.9% of Qf will be added to the fluid

    y = (phi - phi_sc) / phi_cd
    f1 = 6.908 * (m + 1) * (y ** m) * np.exp(-6.908 * y ** (m + 1))
    f1 = np.nan_to_num(f1)

    dmfdphi = f1 * mf_tot / phi_cd

    idx = dmfdphi < 0

    dmfdphi[idx] = 0.0

    return dmfdphi