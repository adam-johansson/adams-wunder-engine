import cantera as ct



def molar_fractions_combustion(T_soc, p_soc, equ_sc, equ_combustion, fuel_type):

    if fuel_type == "CH4":
        # Get all of the Species objects defined in the GRI 3.0 mechanism
        species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}
        # Create an IdealGas object including incomplete combustion species
        #gas2 = ct.Solution(thermo="ideal-gas", species=species.values())

        ohc_species = [species[S] for S in ("CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2", "CH4")]
        gas2 = ct.Solution(thermo="ideal-gas", species=ohc_species)
    elif fuel_type == "jetA":
        # calculate jetA (almost C_12H_23)
        #reaction_mechanism = 'nDodecane_Reitz.yaml'
        species = {S.name: S for S in ct.Species.list_from_file('nDodecane_Reitz.yaml')}
        ohc_species = [species[S] for S in ("co2", "h2o", "o2", "co", "oh", "h2", "o", "h", "n2", "c12h26")]

        # Load the nDodecane mechanism
        #gas2 = ct.Solution(reaction_mechanism)
        gas2 = ct.Solution(thermo="ideal-gas", species=ohc_species)

    elif fuel_type == "H2":
        # calculate H2

        species = {S.name: S for S in ct.Species.list_from_file('h2o2.yaml')}
        ohc_species = [species[S] for S in ("H2O", "O2", "OH", "H2", "O", "H", "N2")]

        # Load the H2 mechanism
        gas2 = ct.Solution(thermo="ideal-gas", species=ohc_species)

    gas2.TP = T_soc, p_soc

    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    N = 5.75 * equ_sc + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

    if fuel_type == "jetA" or "CH4":
        f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
        f2 = 17.75 * (1 - equ_sc)  # O2
        f3 = 12 * equ_sc  # CO2
        f4 = 11.5 * equ_sc  # H2O
        f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

        x_N2 = f1 / N  # molar fractions
        x_O2 = f2 / N
        x_CO2 = f3 / N
        x_H2O = f4 / N
        x_Ar = f5 / N

    elif fuel_type == "H2":
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

    if fuel_type == "CH4":
        gas2.set_equivalence_ratio(equ_combustion, "CH4", f"O2:{x_O2}, N2:{x_N2}, CO2:{x_CO2}, H2O:{x_H2O}")
        gas2.equilibrate("HP")

        fractions = gas2.mole_fraction_dict(threshold=1e-20)

        xi_N2 = fractions["N2"]
        xi_CO2 = fractions["CO2"]
        xi_H2O = fractions["H2O"]
        xi_CO = fractions["CO"]
        xi_O2 = fractions["O2"]
        xi_OH = fractions["OH"]
        xi_H2 = fractions["H2"]
        xi_O = fractions["H"]
        xi_H = fractions["O"]

    elif fuel_type == "jetA":
        gas2.set_equivalence_ratio(equ_combustion, "c12h26:1", f"O2:{x_O2}, N2:{x_N2}, CO2:{x_CO2}, H2O:{x_H2O}")
        gas2.equilibrate("HP")

        fractions = gas2.mole_fraction_dict(threshold=1e-20)

        xi_N2 = fractions["n2"]
        xi_CO2 = fractions["co2"]
        xi_H2O = fractions["h2o"]
        xi_CO = fractions["co"]
        xi_O2 = fractions["o2"]
        xi_OH = fractions["oh"]
        xi_H2 = fractions["h2"]
        xi_O = fractions["h"]
        xi_H = fractions["o"]


    elif fuel_type == "H2":
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



    return xi_N2, xi_CO2, xi_H2O, xi_CO, xi_O2, xi_OH, xi_H2, xi_O, xi_H