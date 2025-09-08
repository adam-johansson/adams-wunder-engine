
import numpy as np
import cantera as ct

from thermo import (
    mixture,
    molar_fractions
)

from thermo.polynomials import N2, O2, O, OH, N, NO, H

from piston_engine.src.piston.nox_integration import improved_nox_integration

# Constants
R_UNIV_J = 8.314510  # J mol^-1 K^-1
R_UNIV_CAL = 1.98720425864083  # cal mol^-1 K^-1
P_STD = 1e5  # Pa
CM3_TO_M3 = 1e-6
PPM_FACTOR = 1e6
G_PER_KG_FACTOR = 1e3
DEFAULT_ODE_MAX_STEP = 1e-5
SPECIES_THRESHOLD = 1e-20


def nox(
        temperatures,
        pressures,
        lambda_z1,
        equ_global,
):


    # Convert arrays to consistent 1D format
    temperatures = np.asarray(temperatures).flatten()
    pressures = np.asarray(pressures).flatten()

    NO_fractions = np.zeros(len(temperatures))

    equ_combustion = 1 / lambda_z1

    fuel_type = "jetA"

    xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0, _ = molar_fractions(
        equivalence_ratio=equ_combustion, fuel_type=fuel_type
    )

    # for the OHC equilibirum
    gas = ct.Solution("gri30.yaml")

    i = 0
    for T,p in zip(temperatures, pressures):
        gas.TPX = (
            T,
            p,
            f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, Ar:{xi_Ar_0}, N2:{xi_N2_0}",
        )

        gas.equilibrate("TP")

        fractions = gas.mole_fraction_dict(threshold=1e-20)

        xi_NO = fractions["NO"]
        NO_fractions[i] = xi_NO
        print(i)
        i = i + 1

    # since we plot PPM with regards to the total outflow mass, we can just use that for all values
    # Get NO molar mass
    t_dummy, p_dummy = 1000.0, 1e5
    _, _, _, _, M_NO = NO(t_dummy, p_dummy)

    _, _, _, _, _, _, _, M_global = mixture(t_dummy, p_dummy, equ_global, fuel_type)

    # Convert to mass fraction
    NO_fractions = NO_fractions * M_NO / M_global

    # Convert to PPM
    NO_fractions = NO_fractions * 1e6

    return NO_fractions
