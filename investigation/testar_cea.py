import CEA_Wrap as cea

from piston_engine.src.piston.thermo_computations_outdated import mixture
from piston_engine.src.piston.fuel_func_outdated import fuel_props


air = cea.Oxidizer("Air", wt_percent=0.98, temp=500)
h2 = cea.Fuel("H2", wt_percent=0.02, temp=500)

problem = cea.TPProblem(materials=[air, h2], temperature=500, pressure=1,
                        pressure_units="bar", temperature_units="k", f_o=1.0, massf=True)

result = problem.run()

print(result.gamma)

far_s, lhv = fuel_props("H2")


equ = 0.02 / far_s

h, u, cp, cv, R, gamma, s = mixture(equ, 500, 1e5, "H2")

print(gamma)