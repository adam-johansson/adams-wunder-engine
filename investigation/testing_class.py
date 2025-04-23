# import pandas as pd
import numpy as np

from CoolProp.CoolProp import PropsSI
from piston_engine.src.piston import polynomials_outdated


# print(coefficients)


# print(coefficients.loc["N2"])

t = 500

from thermo_outdated import PureSubstance

N2 = PureSubstance("N2", temperature=t)
O2 = PureSubstance("O2", temperature=t)
Ar = PureSubstance("Ar", temperature=t)
CO2 = PureSubstance("CO2", temperature=t)
H2O = PureSubstance("H2O", temperature=t)

# N2.temperature = t

# print(N2.heat_capacity(), N2.enthalpy(), N2.entropy(), N2.molar_mass)

# print(polynomials.N2(t, 1e5))

print(O2.coefficients)

p = 1e5
# print(N2.coefficients)
# cpN2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Nitrogen')
# cpO2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Oxygen')
# cpAr_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Argon')
# cpH2O_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'H2O')
# cpCO2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'CO2')


T = np.linspace(300, 6000)

for t in T:
    CO2.temperature = t
    cpCO2_cp = PropsSI("CPMASS", "T", t, "P", p, "CO2")
    print(CO2.heat_capacity(), cpCO2_cp)
