import cantera as ct

species = {S.name: S for S in ct.Species.list_from_file('h2o2.yaml')}
ohc_species = [species[S] for S in ("H2O", "O2", "OH", "H2", "O", "H", "N2")]

# Load the H2 mechanism
gas2 = ct.Solution(thermo="ideal-gas", species=ohc_species)



T = 1000
p = 1e5
equ_sc = 0.0
equ_combustion = 1.0

gas2.TP = T, p

N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
x_O2_air = 1 / N_air  # molar fraction of O2
x_N2_air = 3.7274 / N_air  # molar fraction of N2
x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

N = 0.5 * equ_sc + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
f2 = 0.5 * (1 - equ_sc)  # O2
f4 = equ_sc  # H2O
f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

x_N2 = f1 / N  # molar fractions
x_O2 = f2 / N
x_H2O = f4 / N
x_Ar = f5 / N
x_CO2 = 0.0


gas2.set_equivalence_ratio(equ_combustion, "h2:1", f"O2:{x_O2}, N2:{x_N2}, H2O:{x_H2O}")
gas2.equilibrate("HP")




fractions = gas2.mole_fraction_dict(threshold=1e-20)

xi_N2 = fractions["N2"]
xi_CO2 = 0.0
xi_H2O = fractions["H2O"]
xi_CO = 0.0
xi_O2 = fractions["O2"]
xi_OH = fractions["OH"]
xi_H2 = fractions["H2"]
xi_O = fractions["H"]
xi_H = fractions["O"]