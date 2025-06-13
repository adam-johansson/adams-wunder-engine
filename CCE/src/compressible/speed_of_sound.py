from thermo import mixture
import math


def speed_of_sound(t, equ, fuel_type):
    R_uni = 8.3144626  # J mol^-1 K^-1
    p_dummy = 1e5
    _, _, cp, _, _, _, _, mol = mixture(t, p_dummy, equivalence_ratio=equ, fuel_type=fuel_type)

    R = R_uni / mol
    cv = cp - R
    gamma = cp / cv

    a = math.sqrt(gamma * R * t)

    return a
