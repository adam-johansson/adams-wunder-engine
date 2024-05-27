import numpy as np
from CCE.src.thermo.fluid_props import properties
#from CoolProp.CoolProp import PropsSI


def entropy_func(t, p, equ=0, fuel_type=False):
    R_uni = 8.3144626  # J mol^-1 K^-1
    p_std = 101325  # [Pa] standard pressure
    cp, h, s, M = properties(t, equ, fuel_type)  # get thermo properties for the fluid
    #print(s)
    #s = PropsSI('Smass', 'T', t, 'P', p, 'Air')
    #print(s)
    #M = PropsSI('molarmass', 'T', t, 'P', p, 'Air')
    R = R_uni / M  # specific gas constant


    Psi = (s / R) - np.log(p / p_std)

    return Psi
