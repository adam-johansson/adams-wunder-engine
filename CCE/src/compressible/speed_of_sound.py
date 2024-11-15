from CCE.src import thermo_outdated
import math


def speed_of_sound(t, equ, fuel_type):
    R_uni = 8.3144626  # J mol^-1 K^-1
    p_dummy = 1e5
    cp, h, s, mol = thermo_outdated.properties(t, p_dummy, equ=equ, fuel_type=fuel_type)

    R = R_uni/mol
    cv = cp - R
    gamma = cp/cv

    a = math.sqrt(gamma * R * t)

    return a
