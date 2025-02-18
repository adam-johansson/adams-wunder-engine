from .fuel import fuel_props
from .thermo_computations import mixture, equivalence_derivative, molar_fractions
from .polynomials import N2, O2, Ar, H2, H2O, CO2, JETA, CO
from .work_potential import work_potential
from .entropy_func import entropy_func
from .adiabatic_flame_temperature import flame_temp_inhouse, flame_temp_cea, flame_temp_cantera
from .chemical_equilibrium import equilibrium_OHC
#from .newton_raphson import newton_method
from .ode_solvers import euler, euler_cantera
from .cantera_combustion_fractions import molar_fractions_combustion
