from scipy.optimize import brentq
from thermo import mixture, fuel_props, molar_fractions
import CEA_Wrap as cea
from thermo.polynomials import JETA, H2
import cantera as ct


def flame_temp_inhouse(t_soc, equ_sc, equ_combustion, fuel_type):

    # lower heating value for the fuel
    far_s, lhv = fuel_props(fuel_type)

    # pressure is not effecting enthalpy in NASA polynomials
    p_dummy = 1e5

    # fuel temperature
    t_fuel = 298

    # fuel enthalpy before combustion
    if fuel_type == "jetA":
        cp_f, h_f, s_f, M_f = JETA(t_fuel)
    elif fuel_type == "H2":
        cp_f, h_f, s_f, _, M_f = H2(t_fuel, p_dummy)

    # mixture properties before combustion
    h_soc, _, _, _, _, _, _, _ = mixture(t_soc, p_dummy, equ_sc, fuel_type)

    # fuel air ratio of burned gasses before combustion (CO2 and H2O in mixture)
    far_soc = equ_sc * far_s

    # mass of gas mixture before combustion (excluding fresh fuel)
    m_mix = 1.0

    # mass of pure air in mixture before combustion
    m_air = m_mix / (1 + far_soc)

    # mass of fresh fuel (that is to be burned) before burning
    m_f = far_s * m_air * equ_combustion


    def find_t_flame(t):
        # get enthalpy for stochiometric combustion end products
        h_flame, _, _, _, _, _, _, _ = mixture(t, p_dummy, equ_combustion, fuel_type)

        # energy in the control volume before combustion
        energy_in = h_soc * m_mix + h_f * m_f

        # energy out
        energy_out = h_flame * (m_mix + m_f)

        return energy_out - energy_in

    # find adiabatic flame temperature
    t_flame = brentq(find_t_flame, 200, 6000)

    return t_flame



def flame_temp_cea(t_soc, equ_sc, fuel_type, Psc, equ_combustion):

    # could add fuel temp here
    fueltemp = 298

    # air composition before combustion
    x_N2, x_O2, x_CO2, x_H2O, x_Ar = molar_fractions(equ_sc, fuel_type)

    #air = cea.Oxidizer("Air", temp=t_soc)
    o2 = cea.Oxidizer("O2", temp=t_soc, mols=x_O2)
    n2 = cea.Oxidizer("N2",  temp=t_soc, mols=x_N2)
    h2o = cea.Oxidizer("H2O", temp=t_soc, mols=x_H2O)
    co2 = cea.Oxidizer("CO2", temp=t_soc, mols=x_CO2)

    #print(x_O2, x_CO2, x_H2O, x_N2)

    x_jetA = x_O2 * 17.75 * equ_combustion
    x_H2 = x_O2 * 2.0 * equ_combustion

    jetA = cea.Fuel("Jet-A(L)", temp=fueltemp, mols=x_jetA)

    h2 = cea.Fuel("H2", temp=fueltemp, mols=x_H2)
    if fuel_type == "jetA":
        fuel = jetA
    elif fuel_type == "H2":
        fuel = h2

    # HP problem is like a burner
    #burning = cea.HPProblem(pressure=Psc*1e-5, pressure_units="bar", materials=[n2, o2, h2o, co2, fuel], massf=True,
    #                        phi=equ_combustion)
    burning = cea.HPProblem(pressure=Psc*1e-5, pressure_units="bar", materials=[n2, o2, h2o, co2, fuel], massf=True,
                            phi=equ_combustion)
    exhaust = burning.run()

    t_flame = exhaust.t

    return t_flame



def flame_temp_cantera(T_soc, p_soc, equ_sc, equ_combustion, fuel_type):
    """
    Calculate the flame temperature using Cantera.

    Parameters:
    T_soc (float): Initial temperature in Kelvin.
    p_soc (float): Initial pressure in Pascals.
    equ_sc (float): Equivalence ratio of the gas mixture at start of combustion.
    equ_combustion (float): Equivalence ratio for the combustion process.

    Returns:
    float: Flame temperature in Kelvin.
    """
    try:
        if fuel_type == "CH4":
            # Get all of the Species objects defined in the GRI 3.0 mechanism
            species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

            # Create an IdealGas object including incomplete combustion species
            gas2 = ct.Solution(thermo="ideal-gas", species=species.values())

        elif fuel_type == "jetA":
            species = {S.name: S for S in ct.Species.list_from_file('nDodecane_Reitz.yaml')}
            #reaction_mechanism = 'nDodecane_Reitz.yaml'
            #phase_name = 'nDodecane_IG'

            #
            gas2 = ct.Solution(thermo="ideal-gas", species=species.values())


        gas2.TP = T_soc, p_soc

        N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
        x_O2_air = 1 / N_air  # molar fraction of O2
        x_N2_air = 3.7274 / N_air  # molar fraction of N2
        x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

        N = 5.75 * equ_sc + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

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

        if fuel_type == "CH4":
            gas2.set_equivalence_ratio(equ_combustion, "CH4",
                                       f"O2:{x_O2}, N2:{x_N2}, CO2:{x_CO2}, H2O:{x_H2O}, Ar:{x_Ar}")
        elif fuel_type == "jetA":
            gas2.set_equivalence_ratio(equ_combustion, "c12h26:1", f"O2:{x_O2}, N2:{x_N2}, CO2:{x_CO2}, H2O:{x_H2O}")
        gas2.equilibrate("HP")
        T_flame = gas2.T

        return T_flame
    except Exception as e:
        print(f"An error occurred: {e}")
        return None