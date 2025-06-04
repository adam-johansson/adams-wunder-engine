from thermo.polynomials import N2, O2, CO2, H2O, Ar, JETA_G, H2

# from scipy.optimize import fsolve
from numba import njit


@njit()
def mixture(t, p, equivalence_ratio=0.0, fuel_type="jetA", include_fuel_in_reactants=False, fuel_air_equ_ratio=0.0):
    """
    Calculate thermodynamic properties of combustion gas mixtures.

    This function handles two scenarios:
    1. Pure air that then has fuel added and combusted
    2. Pre-mixed air-fuel mixture that then undergoes combustion

    Parameters
    ----------
    t : float
        Temperature of the mixture [K]
    p : float
        Pressure of the mixture [Pa]
    equivalence_ratio : float
        Equivalence ratio (φ) = (F/A)actual_burned / (F/A)stoichiometric
        This represents how much fuel actually burns, limited by fuel_air_equ_ratio
        - φ = 0: No combustion occurs
        - 0 < φ ≤ fuel_air_equ_ratio: Partial combustion of available fuel
        - Must satisfy: equivalence_ratio ≤ fuel_air_equ_ratio when include_fuel_in_reactants=True
    fuel_type : str
        Type of fuel: "jetA" or "H2"
    include_fuel_in_reactants : bool, optional
        If True, the initial mixture contains fuel mixed with air before combustion.
        If False, pure air is assumed as the starting point. Default is False.
    fuel_air_equ_ratio : float, optional
        When include_fuel_in_reactants=True, this specifies the mass-based
        equivalence ratio of fuel in the initial air-fuel mixture.
        φ_initial = (F/A)_initial / (F/A)_stoichiometric (mass basis). Default is 0.0.

    Returns
    -------
    tuple of floats
        (h, u, cp, cv, R, gamma, s, M_avg)
        - h: specific enthalpy [J/kg]
        - u: specific internal energy [J/kg]
        - cp: specific heat capacity at constant pressure [J/kg·K]
        - cv: specific heat capacity at constant volume [J/kg·K]
        - R: specific gas constant [J/kg·K]
        - gamma: isentropic exponent [-]
        - s: specific entropy [J/kg·K]
        - M_avg: average molar mass [kg/mol]

    Notes
    -----
    Air composition (molar): 21% O2, 78.1% N2, 0.9% Ar
    Stoichiometric reactions:
    - Jet-A: C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
    - H2: H2 + 0.5 O2 → H2O

    Key constraint: When include_fuel_in_reactants=True, equivalence_ratio must be ≤ fuel_air_equ_ratio
    """

    # Validate input constraints
    if include_fuel_in_reactants and equivalence_ratio > fuel_air_equ_ratio:
        raise ValueError(
            f"equivalence_ratio ({equivalence_ratio}) cannot exceed fuel_air_equ_ratio ({fuel_air_equ_ratio}) when fuel is pre-mixed")

    # Get molar masses and properties at actual conditions
    cp_N2, h_N2, _, _, M_N2 = N2(t, p)
    cp_O2, h_O2, _, _, M_O2 = O2(t, p)
    cp_Ar, h_Ar, _, _, M_Ar = Ar(t, p)
    cp_H2O, h_H2O, _, _, M_H2O = H2O(t, p)
    cp_CO2, h_CO2, _, _, M_CO2 = CO2(t, p)
    cp_H2, h_H2, _, _, M_H2 = H2(t, p)
    cp_JETA, h_JETA, _, _, M_JETA = JETA_G(t, p)

    # Air composition (per mole of O2)
    # Standard air: 21% O2, 78.1% N2, 0.9% Ar
    n_N2_per_O2 = 78.1 / 21.0  # ≈ 3.719
    n_Ar_per_O2 = 0.9 / 21.0  # ≈ 0.043

    # Stoichiometric coefficients and mass-based stoichiometric fuel-air ratios
    if fuel_type == "jetA":
        # C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
        nu_O2_stoich = 17.75  # O2 consumed per mole of fuel
        nu_CO2_prod = 12.0  # CO2 produced per mole of fuel
        nu_H2O_prod = 11.5  # H2O produced per mole of fuel
        M_fuel = M_JETA
        cp_fuel = cp_JETA
        h_fuel = h_JETA
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    elif fuel_type == "H2":
        # H2 + 0.5 O2 → H2O
        nu_O2_stoich = 0.5  # O2 consumed per mole of fuel
        nu_CO2_prod = 0.0  # No CO2 produced
        nu_H2O_prod = 1.0  # H2O produced per mole of fuel
        M_fuel = M_H2
        cp_fuel = cp_H2
        h_fuel = h_H2
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Use 'jetA' or 'H2'.")

    # Step 1: Define initial mixture composition
    if include_fuel_in_reactants:
        # Initial mixture is air + fuel based on mass-based equivalence ratio
        # Convert mass-based equivalence ratio to molar fuel content
        # fuel_air_equ_ratio = φ_initial = (F/A)_initial / (F/A)_stoichiometric

        # Basis: 1 mole of O2 in the air component
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_air_total = 1.0 + n_N2_per_O2 + n_Ar_per_O2

        # Mass of air corresponding to this molar amount
        mass_air = n_air_total * M_air

        # Mass of fuel based on equivalence ratio
        mass_fuel_initial = fuel_air_equ_ratio * FA_stoich_mass * mass_air

        # Convert to moles of fuel
        n_fuel_initial = mass_fuel_initial / M_fuel

    else:
        # Initial mixture is pure air
        # Basis: 1 mole of O2
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_fuel_initial = 0.0

    # Initialize final composition
    mu_N2 = mu_O2 = mu_CO2 = mu_H2O = mu_Ar = mu_fuel = 0.0

    # Step 2: Handle combustion
    if equivalence_ratio == 0.0:
        # No combustion - return initial mixture composition
        n_N2_final = n_N2_initial
        n_O2_final = n_O2_initial
        n_CO2_final = 0.0
        n_H2O_final = 0.0
        n_Ar_final = n_Ar_initial
        n_fuel_final = n_fuel_initial

    else:
        # Combustion occurs
        # Determine how much fuel can burn based on equivalence ratio and constraints

        if include_fuel_in_reactants:
            # Fuel is already in the mixture
            # The equivalence_ratio determines what fraction burns (limited by available fuel)
            # Maximum fuel that can burn is limited by available O2
            max_fuel_by_O2 = n_O2_initial / nu_O2_stoich

            # Maximum fuel that can burn is also limited by available fuel
            max_fuel_by_availability = n_fuel_initial

            # The actual amount that can burn based on equivalence_ratio
            # equivalence_ratio is relative to stoichiometric, so we need to consider
            # how much fuel would burn at stoichiometric conditions with available O2
            fuel_at_stoich_with_available_O2 = max_fuel_by_O2

            # Amount we want to burn based on equivalence_ratio
            fuel_target = fuel_at_stoich_with_available_O2 * equivalence_ratio

            # Actual fuel burned is limited by what's available
            n_fuel_burned = min(fuel_target, max_fuel_by_availability, max_fuel_by_O2)

        else:
            # Pure air initially - fuel is added based on equivalence ratio
            # Maximum fuel that can burn is limited by available O2
            max_fuel_by_O2 = n_O2_initial / nu_O2_stoich

            # Amount to burn based on equivalence ratio
            n_fuel_burned = max_fuel_by_O2 * equivalence_ratio

        # Calculate consumption and production
        n_O2_consumed = n_fuel_burned * nu_O2_stoich
        n_CO2_produced = n_fuel_burned * nu_CO2_prod
        n_H2O_produced = n_fuel_burned * nu_H2O_prod

        # Final composition
        n_N2_final = n_N2_initial  # N2 is inert
        n_O2_final = max(0.0, n_O2_initial - n_O2_consumed)
        n_CO2_final = n_CO2_produced
        n_H2O_final = n_H2O_produced
        n_Ar_final = n_Ar_initial  # Ar is inert

        # Unburned fuel
        if include_fuel_in_reactants:
            n_fuel_final = max(0.0, n_fuel_initial - n_fuel_burned)
        else:
            # For pure air case, if equivalence_ratio > 1, there would be excess fuel
            # but since we're not pre-mixing, excess fuel doesn't remain in products
            # (this is a modeling choice - could be changed based on application)
            n_fuel_final = 0.0

    # Step 3: Calculate mass fractions
    # Total moles in final mixture
    n_total = n_N2_final + n_O2_final + n_CO2_final + n_H2O_final + n_Ar_final + n_fuel_final

    if n_total > 0:
        # Molar fractions
        x_N2 = n_N2_final / n_total
        x_O2 = n_O2_final / n_total
        x_CO2 = n_CO2_final / n_total
        x_H2O = n_H2O_final / n_total
        x_Ar = n_Ar_final / n_total
        x_fuel = n_fuel_final / n_total

        # Average molar mass
        M_avg = (x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 +
                 x_H2O * M_H2O + x_Ar * M_Ar + x_fuel * M_fuel)

        # Mass fractions
        mu_N2 = x_N2 * M_N2 / M_avg
        mu_O2 = x_O2 * M_O2 / M_avg
        mu_CO2 = x_CO2 * M_CO2 / M_avg
        mu_H2O = x_H2O * M_H2O / M_avg
        mu_Ar = x_Ar * M_Ar / M_avg
        mu_fuel = x_fuel * M_fuel / M_avg

    # Validation: mass fractions should sum to 1.0
    total_mass_fraction = mu_N2 + mu_O2 + mu_CO2 + mu_H2O + mu_Ar + mu_fuel
    if abs(total_mass_fraction - 1.0) > 1e-10:
        raise RuntimeError(f"Mass fractions sum to {total_mass_fraction}, should be 1.0")

    # Calculate partial pressures for entropy calculation
    p_N2 = x_N2 * p
    p_O2 = x_O2 * p
    p_Ar = x_Ar * p
    p_H2O = x_H2O * p
    p_CO2 = x_CO2 * p
    p_fuel = x_fuel * p

    # Get entropy values using partial pressures
    _, _, s_N2, _, _ = N2(t, p_N2) if p_N2 > 0 else (0, 0, 0, 0, 0)
    _, _, s_O2, _, _ = O2(t, p_O2) if p_O2 > 0 else (0, 0, 0, 0, 0)
    _, _, s_Ar, _, _ = Ar(t, p_Ar) if p_Ar > 0 else (0, 0, 0, 0, 0)
    _, _, s_H2O, _, _ = H2O(t, p_H2O) if p_H2O > 0 else (0, 0, 0, 0, 0)
    _, _, s_CO2, _, _ = CO2(t, p_CO2) if p_CO2 > 0 else (0, 0, 0, 0, 0)

    if fuel_type == "jetA":
        _, _, s_fuel, _, _ = JETA_G(t, p_fuel) if p_fuel > 0 else (0, 0, 0, 0, 0)
    else:
        _, _, s_fuel, _, _ = H2(t, p_fuel) if p_fuel > 0 else (0, 0, 0, 0, 0)

    # Calculate mixture properties
    cp = (
            mu_N2 * cp_N2
            + mu_O2 * cp_O2
            + mu_CO2 * cp_CO2
            + mu_H2O * cp_H2O
            + mu_Ar * cp_Ar
            + mu_fuel * cp_fuel
    )
    h = (
            mu_N2 * h_N2
            + mu_O2 * h_O2
            + mu_CO2 * h_CO2
            + mu_H2O * h_H2O
            + mu_Ar * h_Ar
            + mu_fuel * h_fuel
    )
    s = (
            mu_N2 * s_N2
            + mu_O2 * s_O2
            + mu_CO2 * s_CO2
            + mu_H2O * s_H2O
            + mu_Ar * s_Ar
            + mu_fuel * s_fuel
    )

    Runiv = 8.3144626  # J mol^-1 K^-1
    R = Runiv / M_avg  # specific gas constant
    u = h - R * t  # internal energy
    cv = cp - R  # specific heat capacity at constant volume
    gamma = cp / cv  # isentropic exponent

    return h, u, cp, cv, R, gamma, s, M_avg


@njit()
def equivalence_derivative(equ, t, p, fuel_type, pure_fuel, fuel_equ_ratio):
    """
    Function that returns the partial derivative of specific gas constant R
    and specific internal energy u with respect to the equivalence ratio.

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas. That is fuel air ratio divided by
        stoichiometric fuel air ratio.
    t : float
        Temperature of the gas.
    p : float
        Pressure of the gas.
    fuel_type : string
        "jetA" or "H2".
    pure_fuel : bool
        If True, fuel is present in the initial mixture.
        If False, direct injection (no fuel in initial mixture).
    fuel_equ_ratio : float
        Equivalence ratio of fuel in the initial mixture when pure_fuel=True.

    Returns
    -------
    dellRdellequ : float
        Partial derivative of the specific gas constant R with respect
        to the equivalence ratio.
    delludellequ : float
        Partial derivative of the specific internal energy u with respect
        to the equivalence ratio.
    """

    Runiv = 8.3144626  # J mol^-1 K^-1

    # Air composition (per mole of O2)
    # Standard air: 21% O2, 78.1% N2, 0.9% Ar
    n_N2_per_O2 = 78.1 / 21.0  # ≈ 3.719
    n_Ar_per_O2 = 0.9 / 21.0  # ≈ 0.043
    n_air_per_O2 = 1 + n_N2_per_O2 + n_Ar_per_O2  # total air moles per O2 mole

    # Get molar masses and properties at actual conditions
    _, h_N2, _, _, M_N2 = N2(t, p)
    _, h_O2, _, _, M_O2 = O2(t, p)
    _, h_Ar, _, _, M_Ar = Ar(t, p)
    _, h_H2O, _, _, M_H2O = H2O(t, p)
    _, h_CO2, _, _, M_CO2 = CO2(t, p)
    _, h_H2, _, _, M_H2 = H2(t, p)
    _, h_JETA, _, _, M_JETA = JETA_G(t, p)

    # Define fuel properties and stoichiometric coefficients
    if fuel_type == "jetA":
        # C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
        nu_O2_stoich = 17.75  # O2 consumed per mole of fuel
        nu_CO2_prod = 12.0  # CO2 produced per mole of fuel
        nu_H2O_prod = 11.5  # H2O produced per mole of fuel
        M_fuel = M_JETA
        h_fuel = h_JETA
    elif fuel_type == "H2":
        # H2 + 0.5 O2 → H2O
        nu_O2_stoich = 0.5  # O2 consumed per mole of fuel
        nu_CO2_prod = 0.0  # No CO2 produced
        nu_H2O_prod = 1.0  # H2O produced per mole of fuel
        M_fuel = M_H2
        h_fuel = h_H2
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Use 'jetA' or 'H2'.")

    # Calculate derivatives based on injection type
    if pure_fuel:
        # Fuel is present in initial mixture
        dellRdellequ, delludellequ = _calculate_derivatives_with_fuel_premix(
            equ, t, nu_O2_stoich, nu_CO2_prod, nu_H2O_prod, fuel_equ_ratio,
            n_N2_per_O2, n_Ar_per_O2, n_air_per_O2,
            M_N2, M_O2, M_Ar, M_H2O, M_CO2, M_fuel,
            h_N2, h_O2, h_Ar, h_H2O, h_CO2, h_fuel,
            Runiv
        )
    else:
        # Direct injection - no fuel in initial mixture
        dellRdellequ, delludellequ = _calculate_derivatives_direct_injection(
            equ, t, nu_O2_stoich, nu_CO2_prod, nu_H2O_prod,
            n_N2_per_O2, n_Ar_per_O2, n_air_per_O2,
            M_N2, M_O2, M_Ar, M_H2O, M_CO2,
            h_N2, h_O2, h_Ar, h_H2O, h_CO2,
            Runiv
        )

    return dellRdellequ, delludellequ


@njit()
def _calculate_derivatives_with_fuel_premix(equ, t, nu_O2_stoich, nu_CO2_prod, nu_H2O_prod,
                                            fuel_equ_ratio, n_N2_per_O2, n_Ar_per_O2, n_air_per_O2,
                                            M_N2, M_O2, M_Ar, M_H2O, M_CO2, M_fuel,
                                            h_N2, h_O2, h_Ar, h_H2O, h_CO2, h_fuel, Runiv):
    """Calculate derivatives when fuel is premixed with air."""

    # Total moles and its derivative
    if nu_O2_stoich == 17.75:  # jetA
        delta_moles_per_fuel = nu_CO2_prod + nu_H2O_prod - nu_O2_stoich  # 12 + 11.5 - 17.75 = 5.75
        Ntot = delta_moles_per_fuel * equ + nu_O2_stoich * n_air_per_O2 + fuel_equ_ratio
        dNtotdequ = delta_moles_per_fuel  # 5.75 for jetA
    else:  # H2
        delta_moles_per_fuel = nu_H2O_prod - nu_O2_stoich  # 1 - 0.5 = 0.5
        Ntot = delta_moles_per_fuel * equ + nu_O2_stoich * n_air_per_O2 + fuel_equ_ratio
        dNtotdequ = delta_moles_per_fuel  # 0.5 for H2

    # Moles of each species and their derivatives
    f_N2 = nu_O2_stoich * n_N2_per_O2
    f_O2 = nu_O2_stoich * (1 - equ)
    f_CO2 = nu_CO2_prod * equ
    f_H2O = nu_H2O_prod * equ
    f_Ar = nu_O2_stoich * n_Ar_per_O2
    f_fuel = fuel_equ_ratio - equ

    df_N2_dequ = 0
    df_O2_dequ = -nu_O2_stoich
    df_CO2_dequ = nu_CO2_prod
    df_H2O_dequ = nu_H2O_prod
    df_Ar_dequ = 0
    df_fuel_dequ = -1

    # Molar fractions
    x_N2 = f_N2 / Ntot
    x_O2 = f_O2 / Ntot
    x_CO2 = f_CO2 / Ntot
    x_H2O = f_H2O / Ntot
    x_Ar = f_Ar / Ntot
    x_fuel = f_fuel / Ntot

    # Average molar mass
    M = (x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 +
         x_H2O * M_H2O + x_Ar * M_Ar + x_fuel * M_fuel)

    # Derivatives of molar fractions
    dx_N2_dequ = (df_N2_dequ * Ntot - f_N2 * dNtotdequ) / (Ntot ** 2)
    dx_O2_dequ = (df_O2_dequ * Ntot - f_O2 * dNtotdequ) / (Ntot ** 2)
    dx_CO2_dequ = (df_CO2_dequ * Ntot - f_CO2 * dNtotdequ) / (Ntot ** 2)
    dx_H2O_dequ = (df_H2O_dequ * Ntot - f_H2O * dNtotdequ) / (Ntot ** 2)
    dx_Ar_dequ = (df_Ar_dequ * Ntot - f_Ar * dNtotdequ) / (Ntot ** 2)
    dx_fuel_dequ = (df_fuel_dequ * Ntot - f_fuel * dNtotdequ) / (Ntot ** 2)

    # Derivative of average molar mass
    dM_dequ = (M_N2 * dx_N2_dequ + M_O2 * dx_O2_dequ + M_CO2 * dx_CO2_dequ +
               M_H2O * dx_H2O_dequ + M_Ar * dx_Ar_dequ + M_fuel * dx_fuel_dequ)

    # Derivative of specific gas constant
    dellRdellequ = -(Runiv / M ** 2) * dM_dequ

    # Derivatives of mass fractions
    dmu_N2_dequ = (dx_N2_dequ * M - x_N2 * dM_dequ) * (M_N2 / M ** 2)
    dmu_O2_dequ = (dx_O2_dequ * M - x_O2 * dM_dequ) * (M_O2 / M ** 2)
    dmu_CO2_dequ = (dx_CO2_dequ * M - x_CO2 * dM_dequ) * (M_CO2 / M ** 2)
    dmu_H2O_dequ = (dx_H2O_dequ * M - x_H2O * dM_dequ) * (M_H2O / M ** 2)
    dmu_Ar_dequ = (dx_Ar_dequ * M - x_Ar * dM_dequ) * (M_Ar / M ** 2)
    dmu_fuel_dequ = (dx_fuel_dequ * M - x_fuel * dM_dequ) * (M_fuel / M ** 2)

    # Derivative of specific enthalpy
    dh_dequ = (h_N2 * dmu_N2_dequ + h_O2 * dmu_O2_dequ + h_CO2 * dmu_CO2_dequ +
               h_H2O * dmu_H2O_dequ + h_Ar * dmu_Ar_dequ + h_fuel * dmu_fuel_dequ)

    # Derivative of specific internal energy
    delludellequ = dh_dequ - dellRdellequ * t

    return dellRdellequ, delludellequ


@njit()
def _calculate_derivatives_direct_injection(equ, t, nu_O2_stoich, nu_CO2_prod, nu_H2O_prod,
                                            n_N2_per_O2, n_Ar_per_O2, n_air_per_O2,
                                            M_N2, M_O2, M_Ar, M_H2O, M_CO2,
                                            h_N2, h_O2, h_Ar, h_H2O, h_CO2, Runiv):
    """Calculate derivatives for direct injection (no fuel in initial mixture)."""

    # Total moles and its derivative
    delta_moles_per_fuel = nu_CO2_prod + nu_H2O_prod - nu_O2_stoich
    Ntot = delta_moles_per_fuel * equ + nu_O2_stoich * n_air_per_O2
    dNtotdequ = delta_moles_per_fuel

    # Moles of each species and their derivatives
    f_N2 = nu_O2_stoich * n_N2_per_O2
    f_O2 = nu_O2_stoich * (1 - equ)
    f_CO2 = nu_CO2_prod * equ
    f_H2O = nu_H2O_prod * equ
    f_Ar = nu_O2_stoich * n_Ar_per_O2

    df_N2_dequ = 0
    df_O2_dequ = -nu_O2_stoich
    df_CO2_dequ = nu_CO2_prod
    df_H2O_dequ = nu_H2O_prod
    df_Ar_dequ = 0

    # Molar fractions
    x_N2 = f_N2 / Ntot
    x_O2 = f_O2 / Ntot
    x_CO2 = f_CO2 / Ntot
    x_H2O = f_H2O / Ntot
    x_Ar = f_Ar / Ntot

    # Average molar mass
    M = x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 + x_H2O * M_H2O + x_Ar * M_Ar

    # Derivatives of molar fractions
    dx_N2_dequ = (df_N2_dequ * Ntot - f_N2 * dNtotdequ) / (Ntot ** 2)
    dx_O2_dequ = (df_O2_dequ * Ntot - f_O2 * dNtotdequ) / (Ntot ** 2)
    dx_CO2_dequ = (df_CO2_dequ * Ntot - f_CO2 * dNtotdequ) / (Ntot ** 2)
    dx_H2O_dequ = (df_H2O_dequ * Ntot - f_H2O * dNtotdequ) / (Ntot ** 2)
    dx_Ar_dequ = (df_Ar_dequ * Ntot - f_Ar * dNtotdequ) / (Ntot ** 2)

    # Derivative of average molar mass
    dM_dequ = (M_N2 * dx_N2_dequ + M_O2 * dx_O2_dequ + M_CO2 * dx_CO2_dequ +
               M_H2O * dx_H2O_dequ + M_Ar * dx_Ar_dequ)

    # Derivative of specific gas constant
    dellRdellequ = -(Runiv / M ** 2) * dM_dequ

    # Derivatives of mass fractions
    dmu_N2_dequ = (dx_N2_dequ * M - x_N2 * dM_dequ) * (M_N2 / M ** 2)
    dmu_O2_dequ = (dx_O2_dequ * M - x_O2 * dM_dequ) * (M_O2 / M ** 2)
    dmu_CO2_dequ = (dx_CO2_dequ * M - x_CO2 * dM_dequ) * (M_CO2 / M ** 2)
    dmu_H2O_dequ = (dx_H2O_dequ * M - x_H2O * dM_dequ) * (M_H2O / M ** 2)
    dmu_Ar_dequ = (dx_Ar_dequ * M - x_Ar * dM_dequ) * (M_Ar / M ** 2)

    # Derivative of specific enthalpy
    dh_dequ = (h_N2 * dmu_N2_dequ + h_O2 * dmu_O2_dequ + h_CO2 * dmu_CO2_dequ +
               h_H2O * dmu_H2O_dequ + h_Ar * dmu_Ar_dequ)

    # Derivative of specific internal energy
    delludellequ = dh_dequ - dellRdellequ * t

    return dellRdellequ, delludellequ


@njit()
def molar_fractions(equivalence_ratio, fuel_type, include_fuel_in_reactants=False, fuel_air_ratio=0.0):
    """
    Calculate molar fractions of species in combustion gas mixtures.

    This function handles two scenarios:
    1. Pure air that then has fuel added and combusted
    2. Pre-mixed air-fuel mixture that then undergoes combustion

    Parameters
    ----------
    equivalence_ratio : float
        Equivalence ratio (φ) = (F/A)actual / (F/A)stoichiometric
        - φ = 0: No combustion occurs (pure air or air-fuel mixture)
        - 0 < φ < 1: Lean combustion (excess air)
        - φ = 1: Stoichiometric combustion
        - φ > 1: Rich combustion (excess fuel)
    fuel_type : str
        Type of fuel: "jetA" or "H2"
    include_fuel_in_reactants : bool, optional
        If True, the initial mixture contains fuel mixed with air before combustion.
        If False, pure air is assumed as the starting point. Default is False.
    fuel_air_ratio : float, optional
        When include_fuel_in_reactants=True, this specifies the mass-based
        equivalence ratio of fuel in the initial air-fuel mixture.
        φ_initial = (F/A)_initial / (F/A)_stoichiometric (mass basis). Default is 0.0.

    Returns
    -------
    tuple of floats
        (x_N2, x_O2, x_CO2, x_H2O, x_Ar, x_fuel)
        Molar fractions of N2, O2, CO2, H2O, Ar, and unburned fuel

    Notes
    -----
    Air composition (molar): 21% O2, 78.1% N2, 0.9% Ar
    Stoichiometric reactions:
    - Jet-A: C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
    - H2: H2 + 0.5 O2 → H2O
    """

    # Get molar masses (using dummy values for thermodynamic property calls)
    t_dummy, p_dummy = 999.0, 1e5
    _, _, _, _, M_N2 = N2(t_dummy, p_dummy)
    _, _, _, _, M_O2 = O2(t_dummy, p_dummy)
    _, _, _, _, M_Ar = Ar(t_dummy, p_dummy)
    _, _, _, _, M_H2O = H2O(t_dummy, p_dummy)
    _, _, _, _, M_CO2 = CO2(t_dummy, p_dummy)
    _, _, _, _, M_H2 = H2(t_dummy, p_dummy)
    _, _, _, _, M_JETA = JETA_G(t_dummy, p_dummy)

    # Air composition (per mole of O2)
    # Standard air: 21% O2, 78.1% N2, 0.9% Ar
    n_N2_per_O2 = 78.1 / 21.0  # ≈ 3.719
    n_Ar_per_O2 = 0.9 / 21.0  # ≈ 0.043

    # Stoichiometric coefficients and mass-based stoichiometric fuel-air ratios
    if fuel_type == "jetA":
        # C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
        nu_O2_stoich = 17.75  # O2 consumed per mole of fuel
        nu_CO2_prod = 12.0  # CO2 produced per mole of fuel
        nu_H2O_prod = 11.5  # H2O produced per mole of fuel
        M_fuel = M_JETA
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    elif fuel_type == "H2":
        # H2 + 0.5 O2 → H2O
        nu_O2_stoich = 0.5  # O2 consumed per mole of fuel
        nu_CO2_prod = 0.0  # No CO2 produced
        nu_H2O_prod = 1.0  # H2O produced per mole of fuel
        M_fuel = M_H2
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Use 'jetA' or 'H2'.")

    # Step 1: Define initial mixture composition
    if include_fuel_in_reactants:
        # Initial mixture is air + fuel based on mass-based equivalence ratio
        # Convert mass-based equivalence ratio to molar fuel content
        # fuel_air_ratio = φ_initial = (F/A)_initial / (F/A)_stoichiometric

        # Basis: 1 mole of O2 in the air component
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_air_total = 1.0 + n_N2_per_O2 + n_Ar_per_O2

        # Mass of air corresponding to this molar amount
        mass_air = n_air_total * M_air

        # Mass of fuel based on equivalence ratio
        mass_fuel_initial = fuel_air_ratio * FA_stoich_mass * mass_air

        # Convert to moles of fuel
        n_fuel_initial = mass_fuel_initial / M_fuel

    else:
        # Initial mixture is pure air
        # Basis: 1 mole of O2
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_fuel_initial = 0.0

    # Initialize final composition
    x_N2 = x_O2 = x_CO2 = x_H2O = x_Ar = x_fuel = 0.0

    # Step 2: Handle combustion
    if equivalence_ratio == 0.0:
        # No combustion - return initial mixture composition
        n_N2_final = n_N2_initial
        n_O2_final = n_O2_initial
        n_CO2_final = 0.0
        n_H2O_final = 0.0
        n_Ar_final = n_Ar_initial
        n_fuel_final = n_fuel_initial

    else:
        # Combustion occurs
        # Determine how much fuel can burn based on equivalence ratio and available O2

        if include_fuel_in_reactants:
            # Fuel is already in the mixture
            # The equivalence ratio determines what fraction of the available fuel burns
            max_fuel_that_can_burn = n_O2_initial / nu_O2_stoich

            if equivalence_ratio <= 1.0:
                # Lean/stoichiometric: limited by fuel availability or equivalence ratio
                if n_fuel_initial <= max_fuel_that_can_burn * equivalence_ratio:
                    # All initial fuel participates in combustion scaled by equiv ratio
                    n_fuel_burned = n_fuel_initial * equivalence_ratio
                else:
                    # Limited by equivalence ratio
                    n_fuel_burned = max_fuel_that_can_burn * equivalence_ratio
            else:
                # Rich: limited by available O2
                n_fuel_burned = min(n_fuel_initial, max_fuel_that_can_burn)
        else:
            # Pure air initially - fuel is added based on equivalence ratio
            max_fuel_that_can_burn = n_O2_initial / nu_O2_stoich
            n_fuel_burned = max_fuel_that_can_burn * equivalence_ratio

        # Calculate consumption and production
        n_O2_consumed = n_fuel_burned * nu_O2_stoich
        n_CO2_produced = n_fuel_burned * nu_CO2_prod
        n_H2O_produced = n_fuel_burned * nu_H2O_prod

        # Final composition
        n_N2_final = n_N2_initial  # N2 is inert
        n_O2_final = max(0.0, n_O2_initial - n_O2_consumed)
        n_CO2_final = n_CO2_produced
        n_H2O_final = n_H2O_produced
        n_Ar_final = n_Ar_initial  # Ar is inert

        # Unburned fuel
        if include_fuel_in_reactants:
            n_fuel_final = max(0.0, n_fuel_initial - n_fuel_burned)
        else:
            # In rich conditions, there might be unburned fuel
            if equivalence_ratio > 1.0:
                n_fuel_total_added = n_O2_initial / nu_O2_stoich * equivalence_ratio
                n_fuel_final = max(0.0, n_fuel_total_added - n_fuel_burned)
            else:
                n_fuel_final = 0.0

    # Step 3: Calculate molar fractions
    # Total moles in final mixture
    n_total = n_N2_final + n_O2_final + n_CO2_final + n_H2O_final + n_Ar_final + n_fuel_final

    if n_total > 0:
        # Molar fractions (this is the main difference from mass_fractions function)
        x_N2 = n_N2_final / n_total
        x_O2 = n_O2_final / n_total
        x_CO2 = n_CO2_final / n_total
        x_H2O = n_H2O_final / n_total
        x_Ar = n_Ar_final / n_total
        x_fuel = n_fuel_final / n_total

    # Validation: molar fractions should sum to 1.0
    total_molar_fraction = x_N2 + x_O2 + x_CO2 + x_H2O + x_Ar + x_fuel
    if abs(total_molar_fraction - 1.0) > 1e-10:
        raise RuntimeError(f"Molar fractions sum to {total_molar_fraction}, should be 1.0")

    return x_N2, x_O2, x_CO2, x_H2O, x_Ar, x_fuel


@njit()
def mass_fractions(equivalence_ratio, fuel_type, include_fuel_in_reactants=False, fuel_air_ratio=0.0):
    """
    Calculate mass fractions of species in combustion gas mixtures.

    This function handles two scenarios:
    1. Pure air that then has fuel added and combusted
    2. Pre-mixed air-fuel mixture that then undergoes combustion

    Parameters
    ----------
    equivalence_ratio : float
        Equivalence ratio (φ) = (F/A)actual / (F/A)stoichiometric
        - φ = 0: No combustion occurs (pure air or air-fuel mixture)
        - 0 < φ < 1: Lean combustion (excess air)
        - φ = 1: Stoichiometric combustion
        - φ > 1: Rich combustion (excess fuel)
    fuel_type : str
        Type of fuel: "jetA" or "H2"
    include_fuel_in_reactants : bool, optional
        If True, the initial mixture contains fuel mixed with air before combustion.
        If False, pure air is assumed as the starting point. Default is False.
    fuel_air_ratio : float, optional
        When include_fuel_in_reactants=True, this specifies the mass-based
        equivalence ratio of fuel in the initial air-fuel mixture.
        φ_initial = (F/A)_initial / (F/A)_stoichiometric (mass basis). Default is 0.0.

    Returns
    -------
    tuple of floats
        (mu_N2, mu_O2, mu_CO2, mu_H2O, mu_Ar, mu_fuel)
        Mass fractions of N2, O2, CO2, H2O, Ar, and unburned fuel

    Notes
    -----
    Air composition (molar): 21% O2, 78.1% N2, 0.9% Ar
    Stoichiometric reactions:
    - Jet-A: C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
    - H2: H2 + 0.5 O2 → H2O
    """

    # Get molar masses (using dummy values for thermodynamic property calls)
    t_dummy, p_dummy = 999.0, 1e5
    _, _, _, _, M_N2 = N2(t_dummy, p_dummy)
    _, _, _, _, M_O2 = O2(t_dummy, p_dummy)
    _, _, _, _, M_Ar = Ar(t_dummy, p_dummy)
    _, _, _, _, M_H2O = H2O(t_dummy, p_dummy)
    _, _, _, _, M_CO2 = CO2(t_dummy, p_dummy)
    _, _, _, _, M_H2 = H2(t_dummy, p_dummy)
    _, _, _, _, M_JETA = JETA_G(t_dummy, p_dummy)

    # Air composition (per mole of O2)
    # Standard air: 21% O2, 78.1% N2, 0.9% Ar
    n_N2_per_O2 = 78.1 / 21.0  # ≈ 3.719
    n_Ar_per_O2 = 0.9 / 21.0  # ≈ 0.043

    # Stoichiometric coefficients and mass-based stoichiometric fuel-air ratios
    if fuel_type == "jetA":
        # C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
        nu_O2_stoich = 17.75  # O2 consumed per mole of fuel
        nu_CO2_prod = 12.0  # CO2 produced per mole of fuel
        nu_H2O_prod = 11.5  # H2O produced per mole of fuel
        M_fuel = M_JETA
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    elif fuel_type == "H2":
        # H2 + 0.5 O2 → H2O
        nu_O2_stoich = 0.5  # O2 consumed per mole of fuel
        nu_CO2_prod = 0.0  # No CO2 produced
        nu_H2O_prod = 1.0  # H2O produced per mole of fuel
        M_fuel = M_H2
        # Stoichiometric mass-based fuel-air ratio
        M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
        FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Use 'jetA' or 'H2'.")

    # Step 1: Define initial mixture composition
    if include_fuel_in_reactants:
        # Initial mixture is air + fuel based on mass-based equivalence ratio
        # Convert mass-based equivalence ratio to molar fuel content
        # fuel_air_ratio = φ_initial = (F/A)_initial / (F/A)_stoichiometric

        # Basis: 1 mole of O2 in the air component
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_air_total = 1.0 + n_N2_per_O2 + n_Ar_per_O2

        # Mass of air corresponding to this molar amount
        mass_air = n_air_total * M_air

        # Mass of fuel based on equivalence ratio
        mass_fuel_initial = fuel_air_ratio * FA_stoich_mass * mass_air

        # Convert to moles of fuel
        n_fuel_initial = mass_fuel_initial / M_fuel

    else:
        # Initial mixture is pure air
        # Basis: 1 mole of O2
        n_O2_initial = 1.0
        n_N2_initial = n_N2_per_O2
        n_Ar_initial = n_Ar_per_O2
        n_fuel_initial = 0.0

    # Initialize final composition
    mu_N2 = mu_O2 = mu_CO2 = mu_H2O = mu_Ar = mu_fuel = 0.0

    # Step 2: Handle combustion
    if equivalence_ratio == 0.0:
        # No combustion - return initial mixture composition
        n_N2_final = n_N2_initial
        n_O2_final = n_O2_initial
        n_CO2_final = 0.0
        n_H2O_final = 0.0
        n_Ar_final = n_Ar_initial
        n_fuel_final = n_fuel_initial

    else:
        # Combustion occurs
        # Determine how much fuel can burn based on equivalence ratio and available O2

        if include_fuel_in_reactants:
            # Fuel is already in the mixture
            # The equivalence ratio determines what fraction of the available fuel burns
            max_fuel_that_can_burn = n_O2_initial / nu_O2_stoich

            if equivalence_ratio <= 1.0:
                # Lean/stoichiometric: limited by fuel availability or equivalence ratio
                if n_fuel_initial <= max_fuel_that_can_burn * equivalence_ratio:
                    # All initial fuel participates in combustion scaled by equiv ratio
                    n_fuel_burned = n_fuel_initial * equivalence_ratio
                else:
                    # Limited by equivalence ratio
                    n_fuel_burned = max_fuel_that_can_burn * equivalence_ratio
            else:
                # Rich: limited by available O2
                n_fuel_burned = min(n_fuel_initial, max_fuel_that_can_burn)
        else:
            # Pure air initially - fuel is added based on equivalence ratio
            max_fuel_that_can_burn = n_O2_initial / nu_O2_stoich
            n_fuel_burned = max_fuel_that_can_burn * equivalence_ratio

        # Calculate consumption and production
        n_O2_consumed = n_fuel_burned * nu_O2_stoich
        n_CO2_produced = n_fuel_burned * nu_CO2_prod
        n_H2O_produced = n_fuel_burned * nu_H2O_prod

        # Final composition
        n_N2_final = n_N2_initial  # N2 is inert
        n_O2_final = max(0.0, n_O2_initial - n_O2_consumed)
        n_CO2_final = n_CO2_produced
        n_H2O_final = n_H2O_produced
        n_Ar_final = n_Ar_initial  # Ar is inert

        # Unburned fuel
        if include_fuel_in_reactants:
            n_fuel_final = max(0.0, n_fuel_initial - n_fuel_burned)
        else:
            # In rich conditions, there might be unburned fuel
            if equivalence_ratio > 1.0:
                n_fuel_total_added = n_O2_initial / nu_O2_stoich * equivalence_ratio
                n_fuel_final = max(0.0, n_fuel_total_added - n_fuel_burned)
            else:
                n_fuel_final = 0.0

    # Step 3: Calculate mass fractions
    # Total moles in final mixture
    n_total = n_N2_final + n_O2_final + n_CO2_final + n_H2O_final + n_Ar_final + n_fuel_final

    if n_total > 0:
        # Molar fractions
        x_N2 = n_N2_final / n_total
        x_O2 = n_O2_final / n_total
        x_CO2 = n_CO2_final / n_total
        x_H2O = n_H2O_final / n_total
        x_Ar = n_Ar_final / n_total
        x_fuel = n_fuel_final / n_total

        # Average molar mass
        M_avg = (x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 +
                 x_H2O * M_H2O + x_Ar * M_Ar + x_fuel * M_fuel)

        # Mass fractions
        mu_N2 = x_N2 * M_N2 / M_avg
        mu_O2 = x_O2 * M_O2 / M_avg
        mu_CO2 = x_CO2 * M_CO2 / M_avg
        mu_H2O = x_H2O * M_H2O / M_avg
        mu_Ar = x_Ar * M_Ar / M_avg
        mu_fuel = x_fuel * M_fuel / M_avg

    # Validation: mass fractions should sum to 1.0
    total_mass_fraction = mu_N2 + mu_O2 + mu_CO2 + mu_H2O + mu_Ar + mu_fuel
    if abs(total_mass_fraction - 1.0) > 1e-10:
        raise RuntimeError(f"Mass fractions sum to {total_mass_fraction}, should be 1.0")

    return mu_N2, mu_O2, mu_CO2, mu_H2O, mu_Ar, mu_fuel