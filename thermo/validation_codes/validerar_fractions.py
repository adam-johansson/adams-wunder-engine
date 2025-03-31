from thermo import molar_fractions_combustion, molar_fractions, flame_temp_cantera
import cantera as ct


T_soc = 298
p_soc = 10e5
equ_soc = 0.0
equ_combustion = 1.0

fuel_type = 'jetA'
t_flame = flame_temp_cantera(T_soc, p_soc, equ_soc, equ_combustion, fuel_type)

xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0, _ = molar_fractions(
    equ=equ_combustion, fuel_type=fuel_type
)

# for the OHC equilibirum
gas = ct.Solution("gri30.yaml")

xi_Ar_dummy = xi_Ar_0 + xi_N2_0
# since we want to look at the OHC system isolated, we replace all N2 with Ar
gas.TPX = t_flame, p_soc, f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, Ar:{xi_Ar_dummy}"

gas.equilibrate("TP")

fractions = gas.mole_fraction_dict(threshold=1e-20)

# OHC system (FAST) assuming equilibirum

xi_CO2 = fractions["CO2"]
xi_H2O = fractions["H2O"]
xi_CO = fractions["CO"]
xi_O2 = fractions["O2"]
xi_OH = fractions["OH"]
xi_H2 = fractions["H2"]
xi_O = fractions["H"]
xi_H = fractions["O"]

print(f"Molar fractions from complete combustion (in house):")
print(f"N2: {xi_N2_0}, O2: {xi_O2_0}, CO2: {xi_CO2_0}, H2O:{xi_H2O_0}")

print(f"Molar fractions from OHC system equilibrium after complete combustion (argon) (cantera):")
print(f"CO2: {xi_CO2}, H2O: {xi_H2O}, CO: {xi_CO}, O2:{xi_O2}, OH:{xi_OH}, H2:{xi_H2}, O:{xi_O}, H{xi_H}")


# now do the same but keep the N2 but don't allow it to react
species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

# THIS IS GREAT!!! OHC-system directly!
OHC_species = [species[S] for S in ("CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2")]
gas = ct.Solution(thermo="ideal-gas", species=OHC_species)
gas.TPX = t_flame, p_soc, f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, N2:{xi_N2_0}"
gas.equilibrate("TP")

fractions = gas.mole_fraction_dict(threshold=1e-20)

# OHC system (FAST) assuming equilibirum
xi_N2 = fractions["N2"]
xi_CO2 = fractions["CO2"]
xi_H2O = fractions["H2O"]
xi_CO = fractions["CO"]
xi_O2 = fractions["O2"]
xi_OH = fractions["OH"]
xi_H2 = fractions["H2"]
xi_O = fractions["H"]
xi_H = fractions["O"]

print(f"Molar fractions from OHC system equilibrium after complete combustion (no argon) (cantera):")
print(f"N2: {xi_N2}, CO2: {xi_CO2}, H2O: {xi_H2O}, CO: {xi_CO}, O2: {xi_O2}, OH: {xi_OH}, H2: {xi_H2}, O: {xi_O}, H: {xi_H}")

xi_N2, xi_CO2, xi_H2O, xi_CO, xi_O2, xi_OH, xi_H2, xi_O, xi_H = molar_fractions_combustion(T_soc, p_soc, equ_soc, equ_combustion, fuel_type)

print(f"Molar fractions from Cantera combustion of C12H26:")
print(f"N2: {xi_N2}, CO2: {xi_CO2}, H2O: {xi_H2O}, CO: {xi_CO}, O2: {xi_O2}, OH: {xi_OH}, H2: {xi_H2}, O: {xi_O}, H: {xi_H}")




"""
fuel_type = "CH4"
xi_N2, xi_CO2, xi_H2O, xi_CO, xi_O2, xi_OH, xi_H2, xi_O, xi_H = molar_fractions_combustion(T_soc, p_soc, equ_soc, equ_combustion, fuel_type)

print(f"Molar fractions from Cantera combustion of CH4:")
print(f"N2: {xi_N2}, CO2: {xi_CO2}, H2O: {xi_H2O}, CO: {xi_CO}, O2: {xi_O2}, OH: {xi_OH}, H2: {xi_H2}, O: {xi_O}, H: {xi_H}")
"""
