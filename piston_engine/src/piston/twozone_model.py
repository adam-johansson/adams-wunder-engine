import numpy as np
from scipy import integrate
from thermo import flame_temp_inhouse, flame_temp_cea, mixture, flame_temp_cantera
from numba import njit

# Constants
DEFAULT_LAMBDA_0_DIESEL = 1.00
DEFAULT_LAMBDA_0_H2 = 1.01
DEFAULT_C_FACTOR = 0.15  # for 4 valves and central injection
DEFAULT_POLYTROPE_AVERAGING_DEGREES = 10


def twozone(phi, P, T, V, m, mf, evo, sc, lhv, far_s, equ, fuel_type, factor, premixed):
    """
    Divides the cylinder volume into two zones for more accurate NOx calculations.

    Zone 1: Reaction zone, burned zone (after the flame front)
    Zone 2: Unburned zone, outside the flame front

    Parameters
    ----------
    phi : array_like
        Crank angle [rad]
    P : array_like
        Cylinder pressure [Pa]
    T : array_like
        Cylinder temperature [K]
    V : array_like
        Cylinder volume [m³]
    m : array_like
        Total cylinder mass [kg]
    mf : array_like
        Fuel injection rate [kg/s or kg/rad]
    evo : float
        Exhaust valve opening angle [rad]
    sc : float
        Start of combustion angle [rad]
    lhv : float
        Lower heating value of fuel [J/kg]
    far_s : float
        Stoichiometric fuel-air ratio [-]
    equ : array_like
        Equivalence ratio [-]
    fuel_type : str
        Fuel type ("jetA" or "H2")
    factor : float
        Temperature adjustment factor [-]
    premixed : bool
        Whether the combustion is premixed

    Returns
    -------
    tuple
        (T1, m1, P_hp, V1, lambda_0, phi_hp, equ_hp, T2, m2, T_hp, equ_sc)

    Notes
    -----
    Model assumes both zones have the same pressure: p1 = p2 = p
    """

    # Extract high pressure region (between start of combustion and exhaust valve opening)
    hp_mask = (phi > sc) & (phi < evo)

    phi_hp = phi[hp_mask]
    p_hp = P[hp_mask]
    T_hp = T[hp_mask]
    V_hp = V[hp_mask]
    mf_hp = mf[hp_mask]
    m_hp = m[hp_mask]
    equ_hp = equ[hp_mask]

    # Calculate derived quantities
    qf_hp = mf_hp * lhv  # Heat addition rate

    # Calculate cumulative fuel mass using Simpson's rule
    m_fuel = integrate.cumulative_simpson(mf_hp, x=phi_hp, axis=0, initial=0.0)

    # Global air-fuel equivalence ratio (when all fuel is injected)
    lambda_gl = 1.0 / equ_hp[-1]  # This is now a scalar

    # Determine lambda_0 (air-fuel ratio in reaction zone)
    lambda_0 = _determine_lambda_0(premixed, fuel_type, lambda_gl)

    # Calculate minimum air requirement
    equ_sc = _get_equivalence_ratio_at_start_of_combustion(phi, equ, sc)
    far_sc = equ_sc * far_s

    # Validate to prevent division by zero
    if abs(far_s - far_sc) < 1e-12:
        raise ValueError("Stoichiometric and initial fuel-air ratios are too close")

    L_min = 1.0 / (far_s - far_sc)

    # Calculate zone masses
    m1 = (lambda_0 * L_min + 1) * m_fuel  # Mass in reaction zone
    m2 = m_hp - m1  # Mass in unburned zone

    # Calculate gas constants for both zones
    R1, R2 = _calculate_gas_constants(lambda_0, equ_sc, fuel_type, premixed)

    # Calculate motoring pressure and related quantities
    p_sc, V_sc, T_sc = _get_start_of_combustion_conditions(phi, P, V, T, sc)
    n_poly = _calculate_polytrope_exponent(phi, P, V, sc, DEFAULT_POLYTROPE_AVERAGING_DEGREES)
    p_motor = p_sc * (V_sc / V_hp) ** n_poly
    p_diff = p_hp - p_motor

    # Calculate B function using pressure difference
    integrand = p_diff * m1
    denominator = integrate.simpson(integrand, x=phi_hp, axis=0)

    if abs(denominator) < 1e-12:
        raise ValueError("Denominator in B function calculation is too small")

    nominator = integrate.cumulative_simpson(integrand, x=phi_hp, axis=0, initial=0.0)
    B = 1 - nominator / denominator

    # Calculate adiabatic flame temperature
    t_flame = flame_temp_cea(T_sc, equ_sc, fuel_type, p_sc, 1.0 / lambda_0, premixed=premixed)

    A = (t_flame - T_sc) * factor
    print(f"Flame temp: {t_flame}")
    print(f"Sc temp: {T_sc}")
    Astar = _calculate_astar(A, lambda_gl, lambda_0, premixed, DEFAULT_C_FACTOR)

    # Solve for zone temperatures and volumes
    T1, T2, V1, V2 = _calculate_zone_properties(p_hp, V_hp, m1, m2, R1, R2, Astar, B)

    # Validate volume conservation
    volume_error = np.abs(V_hp - (V1 + V2))
    max_relative_error = np.max(volume_error / V_hp)

    if max_relative_error > 0.01:  # 1% tolerance
        print(f"Warning: Volume conservation error exceeds 1%. Max relative error: {max_relative_error:.3f}")

    return T1, m1, p_hp, V1, lambda_0, phi_hp, equ_hp, T2, m2, T_hp, equ_sc


def _determine_lambda_0(premixed, fuel_type, lambda_gl):
    """Determine the air-fuel ratio in the reaction zone."""
    if premixed:
        return lambda_gl
    elif fuel_type == "H2":
        return DEFAULT_LAMBDA_0_H2
    else:
        return DEFAULT_LAMBDA_0_DIESEL


def _get_equivalence_ratio_at_start_of_combustion(phi, equ, sc):
    """Get equivalence ratio at start of combustion."""
    pre_combustion_mask = phi < sc
    if not np.any(pre_combustion_mask):
        raise ValueError("No data points found before start of combustion")
    return equ[pre_combustion_mask][-1]


def _calculate_gas_constants(lambda_0, equ_sc, fuel_type, premixed):
    """Calculate gas constants for both zones."""
    # Dummy conditions for gas constant calculation (R doesn't depend on T, P)
    t_dummy, p_dummy = 1000.0, 1e5

    # Zone 1: Burned gas at stoichiometric conditions
    _, _, _, _, R1, _, _, _ = mixture(
        t_dummy, p_dummy, equivalence_ratio=1.0 / lambda_0, fuel_type=fuel_type
    )

    # Zone 2: Unburned gas
    if premixed:
        _, _, _, _, R2, _, _, _ = mixture(
            t_dummy, p_dummy,
            equivalence_ratio=equ_sc,
            fuel_type=fuel_type,
            include_fuel_in_reactants=True,
            fuel_air_equ_ratio=1.0 / lambda_0
        )
    else:
        _, _, _, _, R2, _, _, _ = mixture(
            t_dummy, p_dummy, equivalence_ratio=equ_sc, fuel_type=fuel_type
        )

    return R1, R2


def _get_start_of_combustion_conditions(phi, P, V, T, sc):
    """Get pressure and volume at start of combustion."""
    soc_mask = phi > sc
    if not np.any(soc_mask):
        raise ValueError("No data points found after start of combustion")

    p_sc = P[soc_mask][0]
    V_sc = V[soc_mask][0]
    T_sc = T[soc_mask][0]
    return p_sc, V_sc, T_sc


def _calculate_polytrope_exponent(phi, P, V, sc, ca_avg_degrees):
    """Calculate polytrope exponent from data before start of combustion."""
    ca_avg_rad = ca_avg_degrees * np.pi / 180
    poly_mask = (phi > sc - ca_avg_rad) & (phi < sc)

    if np.sum(poly_mask) < 2:
        raise ValueError("Insufficient data points for polytrope exponent calculation")

    P_poly = P[poly_mask]
    V_poly = V[poly_mask]

    # Calculate exponent between consecutive points and take average
    log_P_ratios = np.log(P_poly[1:]) - np.log(P_poly[:-1])
    log_V_ratios = np.log(V_poly[:-1]) - np.log(V_poly[1:])

    # Avoid division by very small numbers
    valid_ratios = np.abs(log_V_ratios) > 1e-12
    if not np.any(valid_ratios):
        raise ValueError("Volume ratios too small for polytrope calculation")

    n_values = log_P_ratios[valid_ratios] / log_V_ratios[valid_ratios]
    return np.mean(n_values)


def _calculate_astar(A, lambda_gl, lambda_0, premixed, C):
    """Calculate the Astar parameter according to Heider's model."""
    if premixed:
        return A
    else:
        if lambda_gl > 1.2:
            return A * (1.2 + (lambda_gl - 1.2) ** C) / (2.2 * lambda_0)
        else:
            return A * 1.2 / (2.2 * lambda_0)


def _calculate_zone_properties(P_hp, V_hp, m1, m2, R1, R2, Astar, B):
    """Calculate temperatures and volumes for both zones."""
    # Zone 1 temperature
    T1 = (P_hp * V_hp + m2 * R2 * Astar * B) / (m1 * R1 + m2 * R2)

    # Zone 2 temperature
    T2 = T1 - Astar * B

    # Zone volumes
    V1 = m1 * R1 * T1 / P_hp
    V2 = m2 * R2 * T2 / P_hp

    return T1, T2, V1, V2