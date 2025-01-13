from scipy.optimize import brentq
from thermo import mixture, fuel_props
import CEA_Wrap as cea
from thermo.polynomials import JETA, H2



def flame_temp_inhouse(t_soc, equ_sc, fuel_type):

    # lower heating value for the fuel
    far_s, lhv = fuel_props(fuel_type)

    # pressure is not effecting enthalpy in NASA polynomials
    p_dummy = 1e5

    # fuel temperature
    t_fuel = 300

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
    m_f = far_s * m_air


    def find_t_flame(t):
        # get enthalpy for stochiometric combustion end products
        h_flame, _, _, _, _, _, _, _ = mixture(t, p_dummy, 1.0, fuel_type)

        # energy in the control volume before combustion
        energy_in = h_soc + m_mix + h_f * m_f

        # energy out
        energy_out =  h_flame * (m_mix + m_f)

        return energy_out - energy_in

    # find adiabatic flame temperature
    t_flame = brentq(find_t_flame, 200, 6000)

    return t_flame



def flame_temp_cea(t_soc, equ_sc, fuel_type, Psc):

    # could add fuel temp here
    fueltemp = 400

    air = cea.Oxidizer("Air", temp=t_soc)

    jetA = cea.Fuel("Jet-A(L)", temp=fueltemp)

    h2 = cea.Fuel("H2", temp=fueltemp)
    if fuel_type == "jetA":
        fuel = jetA
    elif fuel_type == "H2":
        fuel = h2

    # HP problem is like a burner
    burning = cea.HPProblem(pressure=Psc, pressure_units="bar", materials=[air, fuel], massf=True, phi=1.0)
    exhaust = burning.run()

    t_flame = exhaust.t

    return t_flame



