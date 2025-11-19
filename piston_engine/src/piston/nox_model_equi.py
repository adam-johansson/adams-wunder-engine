
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
        masses,
        lambda_z1,
        equ_global,
        m_tot,
):


    # Convert arrays to consistent 1D format
    temperatures = np.asarray(temperatures).flatten()
    pressures = np.asarray(pressures).flatten()
    masses = np.asarray(masses).flatten()

    NO_fractions = np.zeros(len(temperatures))

    equ_combustion = 1 / lambda_z1

    fuel_type = "jetA"

    xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0, _ = molar_fractions(
        equivalence_ratio=equ_combustion, fuel_type=fuel_type
    )

    # for the entire equilibrium
   
    #species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

    #species_names = ["CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2", "NO", "N", "Ar"]

    #ohc_species = [species[name] for name in species_names]
    #gas = ct.Solution(thermo="ideal-gas", species=ohc_species)
    gas = ct.Solution('gri30.yaml')

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
        #print(i)
        i = i + 1

    # since we plot PPM with regards to the total outflow mass, we can just use that for all values
    # Get NO molar mass
    t_dummy, p_dummy = 1000.0, 1e5
    _, _, _, _, M_NO = NO(t_dummy, p_dummy)

    _, _, _, _, _, _, _, M_global = mixture(t_dummy, p_dummy, equ_global, fuel_type)
    _, _, _, _, _, _, _, M_hot_zone = mixture(t_dummy, p_dummy, equ_combustion, fuel_type)

    # Convert to mass fraction inside hot zone
    NO_fractions = NO_fractions * M_NO / M_hot_zone


    # calculate mass of NO (using mass of hot zone)
    m_NO = NO_fractions * masses

    mass_fraction_tot = m_NO / m_tot

    # Convert to PPM
    mass_fraction_tot = mass_fraction_tot * 1e6

    return mass_fraction_tot
