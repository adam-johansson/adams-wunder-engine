import numpy as np
from thermo.thermo_computations import mixture
#from CoolProp.CoolProp import PropsSI


def entropy_func(t, p, equ=0, fuel_type=False):
    #R_uni = 8.3144626  # J mol^-1 K^-1
    p_std = 101325  # [Pa] standard pressure
    h, _, _, _, R, _, s = mixture(t, p, equ)  # get thermo_outdated properties for the fluid
    #print(s)
    #s = PropsSI('Smass', 'T', t, 'P', p, 'Air')
    #print(s)
    #M = PropsSI('molarmass', 'T', t, 'P', p, 'Air')

    Psi = (s / R) - np.log(p / p_std)

    return Psi
