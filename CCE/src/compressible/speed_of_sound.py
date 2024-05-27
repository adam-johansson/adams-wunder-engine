from CCE.src import thermo
import math


def speed_of_sound(t, equ, fuel_type):
    R_uni = 8.3144626  # J mol^-1 K^-1
    cp, h, s, mol = thermo.properties(t, equ=equ, fuel_type=fuel_type)

    R = R_uni/mol
    cv = cp - R
    gamma = cp/cv

    a = math.sqrt(gamma * R * t)

    return a
