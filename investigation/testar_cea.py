import CEA_Wrap as cea

from thermo import mixture, fuel_props
from CCE.src.components import burner

#TP (Specified Temperature and Pressure) Problem Constructor Additional Parameters:
#Very similar to an HP problem, but temperature is specified per-problem and material temperatures are ignored

airtemp = 1300
fueltemp = 400
phicea = 1.0
# all materials must be either fuel or oxidizer

air2 = cea.Oxidizer("Air", temp=airtemp)

jetA = cea.Fuel("Jet-A(L)", temp=fueltemp)

h2 = cea.Fuel("H2", temp=fueltemp)

#o2 = cea.Oxidizer("O2")


# HP problem is like a burner
burning = cea.HPProblem(pressure=10, pressure_units="bar", materials=[air2, jetA], massf=True, phi=phicea)
exhaust = burning.run()
print(exhaust)

print(f"CEA HP problem (burner): cp: {exhaust.cp * 1000}, mw: {exhaust.mw}, gamma: {exhaust.gammas},"
      f" temperature: {exhaust.t}, enthalpy: {exhaust.h}, density: {exhaust.rho}, equ: {phicea}")

p2, t2, f = burner(10e5, airtemp, equ1=0, t2=2190, dp=0.01, eta=0.9999, fuel_type="jetA", t_fuel=fueltemp)

far_s, LHV = fuel_props("jetA")
h, u, cp, cv, R, gamma, s, mw = mixture(t2, p2, f/far_s, "jetA")
rho = p2*R/t2


print(f"My own burner: cp: {cp}, mw: {mw * 1000}, gamma: {gamma},"
      f" temperature: {t2}, enthalpy: {h}, density: {rho}, equ: {f/far_s}")













air = cea.Oxidizer("Air", wt_percent=0.98, temp=500)
h2 = cea.Fuel("H2", wt_percent=0.02, temp=500)

problem = cea.TPProblem(materials=[air, h2], temperature=500, pressure=1,
                        pressure_units="bar", temperature_units="k", f_o=1.0, massf=True)

result = problem.run()

print(result.gamma, result.cp)

far_s, lhv = fuel_props("H2")


equ = 0.02 / far_s

