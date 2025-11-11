import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cantera as ct
from numba import njit

from thermo import (
    mixture,
    molar_fractions,
    equilibrium_OHC,
    euler_cantera,
    molar_fractions_combustion,
)

from thermo.polynomials import N2, O2, O, OH, N, NO, H

from piston_engine.src.piston.nox_integration import improved_nox_integration

# Constants
R_UNIV_J = 8.314510  # J mol^-1 K^-1
R_UNIV_CAL = 1.98720425864083  # cal mol^-1 K^-1
P_STD = 1e5  # Pa
CM3_TO_M3 = 1e-6
PPM_FACTOR = 1e6
G_PER_KG_FACTOR = 1e3
DEFAULT_ODE_MAX_STEP = 1e-5
SPECIES_THRESHOLD = 1e-20


def nox_calculations(
        temperatures,
        pressures,
        volumes,
        fuel_type,
        lambda_z1,
        phi,
        rpm,
        m_tot,
        mf_tot,
        equ_global,
        m_global,
        equ_sc,
):
    """
    Calculate NOx production in the reaction zone using extended Zeldovich mechanism.

    This model calculates NOx produced in the burned zone of a two-zone combustion model
    using the extended Zeldovich mechanism, ignoring NOx creation in flame front and prompt NOx.

    Based on methodology from Merker 2005, "Simulating Combustion".

    Parameters
    ----------
    temperatures : array_like
        Temperature history in burned zone [K]
    masses : array_like
        Mass history in burned zone [kg]
    pressures : array_like
        Pressure history [Pa]
    volumes : array_like
        Volume history in burned zone [m³]
    fuel_type : str
        Fuel type ("jetA" or "H2")
    lambda_z1 : float
        Air-fuel ratio in burned zone [-]
    phi : array_like
        Crank angle [rad]
    rpm : float
        Engine speed [rev/min]
    m_tot : float
        Total mass flowing out during one cycle [kg]
    mf_tot : float
        Total fuel mass injected [kg]
    equ_global : float
        Global equivalence ratio [-]
    m_global : float
        Total cylinder mass [kg]
    equ_sc : float
        Equivalence ratio at start of combustion [-]

    Returns
    -------
    tuple
        (no_concentration_mass, dNOdt_mol, times, EI_nox, m_NO_final)
        - no_concentration_mass: NOx mass concentration [ppm]
        - dNOdt_mol: NOx production rate [mol/s]
        - times: Time vector [s]
        - EI_nox: Emission index [g NOx/kg fuel]
        - m_NO_final: Final NOx mass [kg]
    """

    # Convert arrays to consistent 1D format
    temperatures = np.asarray(temperatures).flatten()
    pressures = np.asarray(pressures).flatten()
    volumes = np.asarray(volumes).flatten()
    phi = np.asarray(phi).flatten()

    # Setup Cantera gas object
    gas = _setup_cantera_gas(fuel_type)

    # Calculate time vector and volume derivatives
    times = _calculate_time_vector(phi, rpm)
    dVdt_s = np.gradient(volumes, times)

    # Get initial conditions and molar fractions
    equ = 1.0 / lambda_z1  # Equivalence ratio in burned zone
    T0, p0 = temperatures[0], pressures[0]

    initial_fractions = molar_fractions_combustion(
        T0, p0, equ_sc=equ_sc, equ_combustion=equ, fuel_type=fuel_type
    )

    # Create ODE function
    def dNOdt_fun(t, var):
        return _nox_ode_function(
            t, var, times, temperatures, pressures, volumes, dVdt_s,
            gas, fuel_type, initial_fractions
        )

    """
    # Solve ODE system
    sol = solve_ivp(
        dNOdt_fun,
        t_span=(times[0], times[-1]),
        method="RK45",
        y0=np.array([0.0]),
        t_eval=times,
        max_step=DEFAULT_ODE_MAX_STEP,
        rtol=1e-8,
        atol=1e-10
    )
    """
    #sol = improved_nox_integration(dNOdt_fun, times)

    sol = solve_ivp(
        dNOdt_fun,
        t_span=(times[0], times[-1]),
        method="LSODA",
        y0=np.array([0.0]),
        t_eval=times,
        max_step=np.inf,
        rtol=1e-6,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Process results
    NO_mol = sol.y[0]

    # Calculate concentrations and emission index
    results = _process_nox_results(
        NO_mol, times, m_tot, mf_tot, equ_global, m_global, fuel_type
    )

    return results


def _setup_cantera_gas(fuel_type):
    """Setup Cantera gas object with appropriate species."""
    try:
        species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}
    except Exception as e:
        raise RuntimeError(f"Failed to load Cantera species: {e}")

    if fuel_type == "jetA":
        species_names = ["CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2"]
    elif fuel_type == "H2":
        species_names = ["H2O", "O2", "OH", "H2", "O", "H", "N2"]
    else:
        raise ValueError(f"Unsupported fuel type: {fuel_type}")

    try:
        ohc_species = [species[name] for name in species_names]
        gas = ct.Solution(thermo="ideal-gas", species=ohc_species)
        return gas
    except KeyError as e:
        raise RuntimeError(f"Missing species in Cantera database: {e}")


def _calculate_time_vector(phi, rpm):
    """Convert crank angles to time."""
    rps = rpm / 60.0
    radians_per_s = rps * 2.0 * np.pi
    return phi / radians_per_s


def _nox_ode_function(t, var, times, temperatures, pressures, volumes, dVdt_s,
                      gas, fuel_type, initial_fractions):
    """ODE function for NOx formation rate."""

    # Find closest time index
    idx = np.argmin(np.abs(times - t))

    # Handle boundary conditions
    idx = max(0, min(idx, len(temperatures) - 1))

    T = temperatures[idx]
    p = pressures[idx]
    V = volumes[idx]
    dVdt = dVdt_s[idx]

    # Current NO concentration
    c_NO = var[0] / V if V > 0 else 0.0

    # Calculate current gas composition
    c_O, c_N2, c_O2, c_OH, c_H = _calculate_gas_concentrations(
        gas, fuel_type, T, p, c_NO, initial_fractions
    )

    # Calculate reaction rate constants
    k1_f, k2_f, k3_f, k1_r, k2_r, k3_r = _calculate_rate_constants(T)

    # Calculate N concentration (quasi-steady state)
    c_N = _calculate_nitrogen_concentration(
        c_O, c_N2, c_O2, c_OH, c_H, c_NO, k1_f, k2_f, k3_f, k1_r, k2_r, k3_r, V, dVdt
    )

    # Calculate NOx formation rate
    dNOdt = _calculate_nox_formation_rate(
        c_O, c_N2, c_N, c_NO, k1_f, k1_r, V, dVdt
    )

    return dNOdt


def _calculate_gas_concentrations(gas, fuel_type, T, p, c_NO, initial_fractions):
    """Calculate gas species concentrations at current conditions."""

    # Unpack initial fractions
    (xi_N2_0, xi_CO2_0, xi_H2O_0, xi_CO_0, xi_O2_0,
     xi_OH_0, xi_H2_0, xi_O_0, xi_H_0) = initial_fractions

    # Account for NO formation in mass balance
    xi_NO = c_NO * (R_UNIV_J * T) / p if p > 0 else 0.0
    xi_O2 = max(0.0, xi_O2_0 - 0.5 * xi_NO)
    xi_N2 = max(0.0, xi_N2_0 - 0.5 * xi_NO)

    # Set gas composition
    try:
        if fuel_type == "jetA":
            composition = (
                f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2}, N2:{xi_N2}, "
                f"CO:{xi_CO_0}, OH:{xi_OH_0}, H2:{xi_H2_0}, O:{xi_O_0}, H:{xi_H_0}"
            )
        else:  # H2
            composition = (
                f"H2O:{xi_H2O_0}, O2:{xi_O2}, N2:{xi_N2}, "
                f"OH:{xi_OH_0}, H2:{xi_H2_0}, O:{xi_O_0}, H:{xi_H_0}"
            )

        gas.TPX = T, p, composition
        gas.equilibrate("TP")

    except Exception as e:
        # Fallback to simplified composition if equilibration fails
        print(f"Warning: Gas equilibration failed at T={T}K, p={p}Pa: {e}")
        return _get_fallback_concentrations(T, p)

    # Extract equilibrium concentrations
    fractions = gas.mole_fraction_dict(threshold=SPECIES_THRESHOLD)

    try:
        xi_O = fractions["O"]
    except:
        xi_O = 0.0
    try:
        xi_H = fractions["H"]
    except:
        xi_H = 0.0
    try:
        xi_O2 = fractions["O2"]
    except:
        xi_O2 = 0.0
    try:
        xi_OH = fractions["OH"]
    except:
        xi_OH = 0.0

    # extract concentrations needed for Zeldovich mechanism
    c_H = (xi_H * p) / (R_UNIV_J * T)
    c_O2 = (xi_O2 * p) / (R_UNIV_J * T)
    c_O = (xi_O * p) / (R_UNIV_J * T)
    c_OH = (xi_OH * p) / (R_UNIV_J * T)
    c_N2 = (xi_N2 * p) / (R_UNIV_J * T)

    return c_O, c_N2, c_O2, c_OH, c_H


def _get_fallback_concentrations(T, p):
    """Provide fallback concentrations if equilibration fails."""
    # Simple approximation - could be improved with more sophisticated fallback
    return {
        'c_O': 1e-10,
        'c_H': 1e-10,
        'c_O2': 0.01 * p / (R_UNIV_J * T),
        'c_OH': 1e-6,
        'c_N2': 0.7 * p / (R_UNIV_J * T)
    }


@njit()
def _calculate_rate_constants(T):
    """Calculate forward and reverse reaction rate constants."""

    # Get thermodynamic data for equilibrium constants
    _, _, _, g_N2, M_N2 = N2(T, P_STD)
    _, _, _, g_O, M_O = O(T, P_STD)
    _, _, _, g_NO, M_NO = NO(T, P_STD)
    _, _, _, g_N, M_N = N(T, P_STD)
    _, _, _, g_O2, M_O2 = O2(T, P_STD)
    _, _, _, g_OH, M_OH = OH(T, P_STD)
    _, _, _, g_H, M_H = H(T, P_STD)

    g_N2 = g_N2 * M_N2  # Convert to molar basis
    g_O = g_O * M_O  # Convert to molar basis
    g_NO = g_NO * M_NO  # Convert to molar basis
    g_N = g_N * M_N  # Convert to molar basis
    g_O2 = g_O2 * M_O2  # Convert to molar basis
    g_OH = g_OH * M_OH  # Convert to molar basis
    g_H = g_H * M_H  # Convert to molar basis

    # Calculate equilibrium constants
    Delta_g_R_1 = g_NO + g_N - g_N2 - g_O
    Delta_g_R_2 = g_NO + g_O - g_N - g_O2
    Delta_g_R_3 = g_NO + g_H - g_N - g_OH

    Kc_1 = math.exp(-Delta_g_R_1 / (R_UNIV_J * T))
    Kc_2 = math.exp(-Delta_g_R_2 / (R_UNIV_J * T))
    Kc_3 = math.exp(-Delta_g_R_3 / (R_UNIV_J * T))

    # GRI-Mech 3.0 rate constants

    # Forward rate constants (convert from cm³ to m³)
    # Note: GRI-Mech reaction (1) is defined as N+NO<=>N2+O, so we use reverse coefficients
    k1_r = 2.70e13 * math.exp(-355.0 / (R_UNIV_CAL * T)) * CM3_TO_M3
    k2_f = 9.0e9 * T * math.exp(-6500.0 / (R_UNIV_CAL * T)) * CM3_TO_M3
    k3_f = 3.36e13 * math.exp(-385.0 / (R_UNIV_CAL * T)) * CM3_TO_M3

    # Calculate other rate constants using equilibrium relationships
    k1_f = k1_r * Kc_1
    k2_r = k2_f / Kc_2
    k3_r = k3_f / Kc_3

    return k1_f, k2_f, k3_f, k1_r, k2_r, k3_r


@njit()
def _calculate_nitrogen_concentration(c_O, c_N2, c_O2, c_OH, c_H, c_NO, k1_f, k2_f, k3_f, k1_r, k2_r, k3_r, V, dVdt):
    """Calculate N atom concentration using quasi-steady state assumption."""

    # Quasi-steady state: dc_N/dt = 0
    numerator = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H)
    #KOLLA UPP DETTA!

    if V > 0:
        denominator = k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH + dVdt / V
    else:
        denominator = k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH

    # Remove dVdt/V term - quasi-steady state based on chemistry only
    #denominator = k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH

    # Prevent division by very small numbers
    if abs(denominator) < 1e-20:
        return 0.0

    return numerator / denominator


@njit()
def _calculate_nox_formation_rate(c_O, c_N2, c_N, c_NO, k1_f, k1_r, V, dVdt):
    """Calculate the net NOx formation rate."""

    # Net rate from extended Zeldovich mechanism (primarily reaction 1)
    dc_NOdt_chem = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N

    if V > 0:
        # Include volume change effects
        dc_NOdt_vol = -(c_NO + c_N) * dVdt / V
        dc_NOdt = dc_NOdt_chem + dc_NOdt_vol
        # Convert concentration rate to molar rate
        dNOdt = dc_NOdt * V + c_NO * dVdt
    else:
        if V >= 0:
            dNOdt = dc_NOdt_chem * V
        else:
            dNOdt = 0.0

    return dNOdt


@njit()
def _process_nox_results(NO_mol, times, m_tot, mf_tot, equ_global, m_global, fuel_type):
    """Process NOx calculation results and calculate concentrations."""

    # Calculate gradient
    dNOdt_mol = numba_gradient_uniform(NO_mol, times)

    # Get NO molar mass
    t_dummy, p_dummy = 1000.0, 1e5
    _, _, _, _, M_NO = NO(t_dummy, p_dummy)

    # Convert to mass
    m_NO = NO_mol * M_NO

    # Mass-based concentration (ppm)
    no_concentration_mass = (m_NO / m_tot) * PPM_FACTOR

    # Volume-based concentration (ppm)
    _, _, _, _, _, _, _, M_global = mixture(t_dummy, p_dummy, equ_global, fuel_type)
    mol_global = m_global / M_global
    no_concentration_volume = (NO_mol / mol_global) * PPM_FACTOR

    # Emission index (g NOx/kg fuel)
    EI_nox = (m_NO[-1] / mf_tot) * G_PER_KG_FACTOR

    #return (no_concentration_mass, dNOdt_mol, times, EI_nox, m_NO[-1])
    return (no_concentration_mass, dNOdt_mol, times, EI_nox, m_NO)


@njit
def numba_gradient_uniform(y, x):
    """
    Simple Numba-compatible gradient for uniform spacing.

    Parameters
    ----------
    y : array_like
        Input array for which gradient is calculated
    x : array_like
        Sample points (assumes uniform spacing)

    Returns
    -------
    gradient : ndarray
        Gradient of y with respect to x
    """
    n = len(y)
    gradient = np.zeros_like(y)

    # Calculate uniform spacing
    dx = x[1] - x[0]

    # Forward difference at start
    gradient[0] = (y[1] - y[0]) / dx

    # Central difference in middle
    for i in range(1, n - 1):
        gradient[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx)

    # Backward difference at end
    gradient[n - 1] = (y[n - 1] - y[n - 2]) / dx

    return gradient