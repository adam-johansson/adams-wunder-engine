from thermo.thermo_computations import mixture



def entropy_func(t, p, equ=0, fuel_type=False):

    if fuel_type:
        _, _, _, _, R, _, s, _ = mixture(
            t, p, equ, fuel_type
        )
    else:
        _, _, _, _, R, _, s, _ = mixture(
            t, p, equ
        )


    # the entropy is already corrected for pressure in the polynomials functions
    Psi = (s / R) #- np.log(p / p_std)

    return Psi
