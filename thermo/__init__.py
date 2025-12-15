from .fuel import fuel_props
from .thermo_computations import (
    mixture,
    equivalence_derivative,
    molar_fractions,
    mass_fractions,
)
from .polynomials import N2, O2, Ar, H2, H2O, CO2, JETA_L, JETA_G, CO
from .polynomials_vector import (N2_vectorized, O2_vectorized, Ar_vectorized, H2_vectorized,
                                 H2O_vectorized, CO2_vectorized, JETA_G_vectorized, JETA_L_vectorized)
from .work_potential import work_potential
from .entropy_func import entropy_func
from .adiabatic_flame_temperature import (
    flame_temp_inhouse,
    flame_temp_cea,
    flame_temp_cantera,
)
from .chemical_equilibrium import equilibrium_OHC

# from .newton_raphson import newton_method
from .ode_solvers import euler, euler_cantera
from .cantera_combustion_fractions import molar_fractions_combustion

from.thermo_computations_vector import mixture_vectorized

