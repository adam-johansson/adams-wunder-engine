import torch
from typing import Tuple

# Import the vectorized NASA polynomials
from thermo import (
    N2_vectorized, O2_vectorized, CO2_vectorized, H2O_vectorized,
    Ar_vectorized, JETA_G_vectorized, H2_vectorized
)


def mixture_vectorized(
        t: torch.Tensor,
        p: torch.Tensor,
        equivalence_ratio: torch.Tensor = None,
        fuel_type: str = "jetA",
        include_fuel_in_reactants: bool = False,
        fuel_air_equ_ratio: torch.Tensor = None
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized calculation of thermodynamic properties of combustion gas mixtures.

    This function handles two scenarios:
    1. Pure air that then has fuel added and combusted
    2. Pre-mixed air-fuel mixture that then undergoes combustion

    Parameters
    ----------
    t : torch.Tensor
        Temperature of the mixture [K], shape (batch_size,)
    p : torch.Tensor
        Pressure of the mixture [Pa], shape (batch_size,)
    equivalence_ratio : torch.Tensor, optional
        Equivalence ratio (φ) = (F/A)actual_burned / (F/A)stoichiometric, shape (batch_size,)
        Default is tensor of zeros
    fuel_type : str
        Type of fuel: "jetA" or "H2"
    include_fuel_in_reactants : bool, optional
        If True, the initial mixture contains fuel mixed with air before combustion.
        If False, pure air is assumed as the starting point. Default is False.
    fuel_air_equ_ratio : torch.Tensor, optional
        When include_fuel_in_reactants=True, this specifies the mass-based
        equivalence ratio of fuel in the initial air-fuel mixture, shape (batch_size,)
        Default is tensor of zeros

    Returns
    -------
    Tuple of torch.Tensor
        (h, u, cp, cv, R, gamma, s, M_avg) each with shape (batch_size,)
        - h: specific enthalpy [J/kg]
        - u: specific internal energy [J/kg]
        - cp: specific heat capacity at constant pressure [J/kg·K]
        - cv: specific heat capacity at constant volume [J/kg·K]
        - R: specific gas constant [J/kg·K]
        - gamma: isentropic exponent [-]
        - s: specific entropy [J/kg·K]
        - M_avg: average molar mass [kg/mol]
    """
    batch_size = t.shape[0]
    device = t.device
    dtype = t.dtype

    # Default values for optional parameters
    if equivalence_ratio is None:
        equivalence_ratio = torch.zeros(batch_size, device=device, dtype=dtype)
    if fuel_air_equ_ratio is None:
        fuel_air_equ_ratio = torch.zeros(batch_size, device=device, dtype=dtype)

    # Get properties at actual conditions
    cp_N2, h_N2, s_N2_std, _, M_N2 = N2_vectorized(t, p)
    cp_O2, h_O2, s_O2_std, _, M_O2 = O2_vectorized(t, p)
    cp_Ar, h_Ar, s_Ar_std, _, M_Ar = Ar_vectorized(t, p)
    cp_H2O, h_H2O, s_H2O_std, _, M_H2O = H2O_vectorized(t, p)
    cp_CO2, h_CO2, s_CO2_std, _, M_CO2 = CO2_vectorized(t, p)
    cp_H2, h_H2, s_H2_std, _, M_H2 = H2_vectorized(t, p)
    cp_JETA, h_JETA, s_JETA_std, _, M_JETA = JETA_G_vectorized(t, p)

    # Air composition (per mole of O2)
    n_N2_per_O2 = 78.1 / 21.0  # ≈ 3.719
    n_Ar_per_O2 = 0.9 / 21.0  # ≈ 0.043

    # Stoichiometric coefficients and mass-based stoichiometric fuel-air ratios
    if fuel_type == "jetA":
        # C12H23 + 17.75 O2 → 12 CO2 + 11.5 H2O
        nu_O2_stoich = 17.75
        nu_CO2_prod = 12.0
        nu_H2O_prod = 11.5
        M_fuel = M_JETA
        cp_fuel = cp_JETA
        h_fuel = h_JETA
        s_fuel_std = s_JETA_std
    elif fuel_type == "H2":
        # H2 + 0.5 O2 → H2O
        nu_O2_stoich = 0.5
        nu_CO2_prod = 0.0
        nu_H2O_prod = 1.0
        M_fuel = M_H2
        cp_fuel = cp_H2
        h_fuel = h_H2
        s_fuel_std = s_H2_std
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Use 'jetA' or 'H2'.")

    # Average molar mass of air
    M_air = (M_O2 + n_N2_per_O2 * M_N2 + n_Ar_per_O2 * M_Ar) / (1 + n_N2_per_O2 + n_Ar_per_O2)
    FA_stoich_mass = M_fuel / (nu_O2_stoich * M_air * (1 + n_N2_per_O2 + n_Ar_per_O2))

    # Step 1: Define initial mixture composition
    if include_fuel_in_reactants:
        # Initial mixture is air + fuel based on mass-based equivalence ratio
        n_O2_initial = torch.ones(batch_size, device=device, dtype=dtype)
        n_N2_initial = torch.full((batch_size,), n_N2_per_O2, device=device, dtype=dtype)
        n_Ar_initial = torch.full((batch_size,), n_Ar_per_O2, device=device, dtype=dtype)
        n_air_total = 1.0 + n_N2_per_O2 + n_Ar_per_O2

        # Mass of air and fuel
        mass_air = n_air_total * M_air
        mass_fuel_initial = fuel_air_equ_ratio * FA_stoich_mass * mass_air
        n_fuel_initial = mass_fuel_initial / M_fuel
    else:
        # Initial mixture is pure air
        n_O2_initial = torch.ones(batch_size, device=device, dtype=dtype)
        n_N2_initial = torch.full((batch_size,), n_N2_per_O2, device=device, dtype=dtype)
        n_Ar_initial = torch.full((batch_size,), n_Ar_per_O2, device=device, dtype=dtype)
        n_fuel_initial = torch.zeros(batch_size, device=device, dtype=dtype)

    # Step 2: Handle combustion
    no_combustion = (equivalence_ratio == 0.0)

    # Initialize final composition
    n_N2_final = n_N2_initial.clone()
    n_O2_final = n_O2_initial.clone()
    n_CO2_final = torch.zeros(batch_size, device=device, dtype=dtype)
    n_H2O_final = torch.zeros(batch_size, device=device, dtype=dtype)
    n_Ar_final = n_Ar_initial.clone()
    n_fuel_final = n_fuel_initial.clone()

    # Handle combustion where equivalence_ratio > 0
    combustion_mask = ~no_combustion

    if combustion_mask.any():
        if include_fuel_in_reactants:
            # Fuel is already in the mixture
            max_fuel_by_O2 = n_O2_initial / nu_O2_stoich
            max_fuel_by_availability = n_fuel_initial
            fuel_target = max_fuel_by_O2 * equivalence_ratio

            n_fuel_burned = torch.minimum(
                torch.minimum(fuel_target, max_fuel_by_availability),
                max_fuel_by_O2
            )
        else:
            # Pure air initially - fuel is added based on equivalence ratio
            max_fuel_by_O2 = n_O2_initial / nu_O2_stoich
            n_fuel_burned = max_fuel_by_O2 * equivalence_ratio

        # Apply combustion only where needed
        n_fuel_burned = torch.where(combustion_mask, n_fuel_burned,
                                    torch.zeros_like(n_fuel_burned))

        # Calculate consumption and production
        n_O2_consumed = n_fuel_burned * nu_O2_stoich
        n_CO2_produced = n_fuel_burned * nu_CO2_prod
        n_H2O_produced = n_fuel_burned * nu_H2O_prod

        # Update final composition
        n_O2_final = torch.clamp(n_O2_initial - n_O2_consumed, min=0.0)
        n_CO2_final = n_CO2_produced
        n_H2O_final = n_H2O_produced

        # Unburned fuel
        if include_fuel_in_reactants:
            n_fuel_final = torch.clamp(n_fuel_initial - n_fuel_burned, min=0.0)
        else:
            n_fuel_final = torch.zeros(batch_size, device=device, dtype=dtype)

    # Step 3: Calculate mass fractions
    n_total = n_N2_final + n_O2_final + n_CO2_final + n_H2O_final + n_Ar_final + n_fuel_final

    # Avoid division by zero
    eps = torch.finfo(dtype).eps
    n_total = torch.clamp(n_total, min=eps)

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

    # Calculate partial pressures for entropy calculation
    p_N2 = x_N2 * p
    p_O2 = x_O2 * p
    p_Ar = x_Ar * p
    p_H2O = x_H2O * p
    p_CO2 = x_CO2 * p
    p_fuel = x_fuel * p

    # Get entropy values using partial pressures
    # Only calculate where partial pressure > 0
    s_N2 = torch.where(p_N2 > 0, N2_vectorized(t, p_N2)[2], torch.zeros_like(t))
    s_O2 = torch.where(p_O2 > 0, O2_vectorized(t, p_O2)[2], torch.zeros_like(t))
    s_Ar = torch.where(p_Ar > 0, Ar_vectorized(t, p_Ar)[2], torch.zeros_like(t))
    s_H2O = torch.where(p_H2O > 0, H2O_vectorized(t, p_H2O)[2], torch.zeros_like(t))
    s_CO2 = torch.where(p_CO2 > 0, CO2_vectorized(t, p_CO2)[2], torch.zeros_like(t))

    if fuel_type == "jetA":
        s_fuel = torch.where(p_fuel > 0, JETA_G_vectorized(t, p_fuel)[2], torch.zeros_like(t))
    else:
        s_fuel = torch.where(p_fuel > 0, H2_vectorized(t, p_fuel)[2], torch.zeros_like(t))

    # Calculate mixture properties
    cp = (mu_N2 * cp_N2 + mu_O2 * cp_O2 + mu_CO2 * cp_CO2 +
          mu_H2O * cp_H2O + mu_Ar * cp_Ar + mu_fuel * cp_fuel)

    h = (mu_N2 * h_N2 + mu_O2 * h_O2 + mu_CO2 * h_CO2 +
         mu_H2O * h_H2O + mu_Ar * h_Ar + mu_fuel * h_fuel)

    s = (mu_N2 * s_N2 + mu_O2 * s_O2 + mu_CO2 * s_CO2 +
         mu_H2O * s_H2O + mu_Ar * s_Ar + mu_fuel * s_fuel)

    # Gas constants and derived properties
    Runiv = torch.tensor(8.3144626, device=device, dtype=dtype)  # J mol^-1 K^-1
    R = Runiv / M_avg  # specific gas constant
    u = h - R * t  # internal energy
    cv = cp - R  # specific heat capacity at constant volume
    gamma = cp / cv  # isentropic exponent

    return h, u, cp, cv, R, gamma, s, M_avg
