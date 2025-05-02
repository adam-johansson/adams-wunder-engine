import numpy as np
from thermo.thermo_computations import mixture

# from CoolProp.CoolProp import PropsSI


def entropy_func(t, p, equ=0, fuel_type=False):

    _, _, _, _, R, _, s, _ = mixture(
        t, p, equ, fuel_type
    )


    # the entropy is already corrected for pressure in the polynomials functions
    Psi = (s / R) #- np.log(p / p_std)

    return Psi
