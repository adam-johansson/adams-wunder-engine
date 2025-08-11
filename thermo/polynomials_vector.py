import torch

# Universal gas constant from NASA polynomials pdf
R = 8.314510  # J mol^-1 K^-1


def N2_vectorized(T, p):
    """Vectorized N2 polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)  # clamp temperature
    M = torch.tensor(28.0134e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    # Coefficient arrays
    a_low = torch.tensor([
        2.210371497e04, -3.818461820e02, 6.082738360e00,
        -8.530914410e-03, 1.384646189e-05, -9.625793620e-09, 2.519705809e-12,
        7.108460860e02, -1.076003744e01,
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        5.877124060e05, -2.239249073e03, 6.066949220e00,
        -6.139685500e-04, 1.491806679e-07, -1.923105485e-11, 1.061954386e-15,
        1.283210415e04, -1.586640027e01,
    ], device=T.device, dtype=T.dtype)

    # Choose coefficients without branching
    use_low = (T < 1000.0007).float().unsqueeze(1)  # Shape: (batch, 1)
    a = use_low * a_low + (1.0 - use_low) * a_high  # Broadcasting

    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T  # Unpack coefficients

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def O2_vectorized(T, p):
    """Vectorized O2 polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)
    M = torch.tensor(31.9988e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    # Low temperature coefficients (200K-1000K)
    a_low = torch.tensor([
        -3.425563420e04, 4.847000970e02, 1.119010961e00,
        4.293889240e-03, -6.836300520e-07, -2.023372700e-09, 1.039040018e-12,
        -3.391454870e03, 1.849699470e01
    ], device=T.device, dtype=T.dtype)

    # High temperature coefficients (1000K-6000K)
    a_high = torch.tensor([
        -1.037939022e06, 2.344830282e03, 1.819732036e00,
        1.267847582e-03, -2.188067988e-07, 2.053719572e-11, -8.193467050e-16,
        -1.689010929e04, 1.738716506e01
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def Ar_vectorized(T, p):
    """Vectorized Ar polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)
    M = torch.tensor(39.948e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a_low = torch.tensor([
        0.0, 0.0, 2.500000000e00, 0.0, 0.0, 0.0, 0.0,
        -7.453750000e02, 4.379674910e00
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        2.010538475e01, -5.992661070e-02, 2.500069401e00,
        -3.992141160e-08, 1.205272140e-11, -1.819015576e-15, 1.078576636e-19,
        -7.449939610e02, 4.379180110e00
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def CO2_vectorized(T, p):
    """Vectorized CO2 polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)
    M = torch.tensor(44.0095e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a_low = torch.tensor([
        4.943650540e04, -6.264116010e02, 5.301725240e00,
        2.503813816e-03, -2.127308728e-07, -7.689988780e-10, 2.849677801e-13,
        -4.528198460e04, -7.048279440e00
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        1.176962419e05, -1.788791477e03, 8.291523190e00,
        -9.223156780e-05, 4.863676880e-09, -1.891053312e-12, 6.330036590e-16,
        -3.908350590e04, -2.652669281e01
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def H2O_vectorized(T, p):
    """Vectorized H2O polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)
    M = torch.tensor(18.01528e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a_low = torch.tensor([
        -3.947960830e04, 5.755731020e02, 9.317826530e-01,
        7.222712860e-03, -7.342557370e-06, 4.955043490e-09, -1.336933246e-12,
        -3.303974310e04, 1.724205775e01
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        1.034972096e06, -2.412698562e03, 4.646110780e00,
        2.291998307e-03, -6.836830480e-07, 9.426468930e-11, -4.822380530e-15,
        -1.384286509e04, -7.978148510e00
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def JETA_L_vectorized(T):
    """Vectorized JETA_L polynomial for PyTorch tensors"""
    # Check temperature range (you might want to handle this differently)
    T = torch.clamp(T, 220.0, 550.0)

    M = torch.tensor(167.31102e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a1 = torch.tensor(-4.218262130e05, device=T.device, dtype=T.dtype)
    a2 = torch.tensor(-5.576600450e03, device=T.device, dtype=T.dtype)
    a3 = torch.tensor(1.522120958e02, device=T.device, dtype=T.dtype)
    a4 = torch.tensor(-8.610197550e-01, device=T.device, dtype=T.dtype)
    a5 = torch.tensor(3.071662234e-03, device=T.device, dtype=T.dtype)
    a6 = torch.tensor(-4.702789540e-06, device=T.device, dtype=T.dtype)
    a7 = torch.tensor(2.743019833e-09, device=T.device, dtype=T.dtype)
    b1 = torch.tensor(-3.238369150e04, device=T.device, dtype=T.dtype)
    b2 = torch.tensor(-6.781094910e02, device=T.device, dtype=T.dtype)

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (
            a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4
    )

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT + a4 * T +
            a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    # Enthalpy of formation (set to 0 as in original)
    hf = torch.tensor(0.0, device=T.device, dtype=T.dtype)
    h = h - hf

    return cp, h, s, M


def JETA_G_vectorized(T, p):
    """Vectorized JETA_G polynomial for PyTorch tensors"""
    T = torch.clamp(T, 273.15, 6000.0)
    M = torch.tensor(167.31102e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a_low = torch.tensor([
        -6.068695590e05, 8.328259590e03, -4.312321270e01,
        2.572390455e-01, -2.629316040e-04, 1.644988940e-07, -4.645335140e-11,
        -7.606962760e04, 2.794305937e02
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        1.858356102e07, -7.677219890e04, 1.419826133e02,
        -7.437524530e-03, 5.856202550e-07, 1.223955647e-11, -3.149201922e-15,
        4.221989520e05, -8.986061040e02
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M


def H2_vectorized(T, p):
    """Vectorized H2 polynomial for PyTorch tensors"""
    T = torch.clamp(T, 200.0, 6000.0)
    M = torch.tensor(2.0158800e-3, device=T.device, dtype=T.dtype)
    Rspec = R / M

    a_low = torch.tensor([
        4.078323210e04, -8.009186040e02, 8.214702010e00,
        -1.269714457e-02, 1.753605076e-05, -1.202860270e-08, 3.368093490e-12,
        2.682484665e03, -3.043788844e01
    ], device=T.device, dtype=T.dtype)

    a_high = torch.tensor([
        5.608128010e05, -8.371504740e02, 2.975364532e00,
        1.252249124e-03, -3.740716190e-07, 5.936625200e-11, -3.606994100e-15,
        5.339824410e03, -2.202774769e00
    ], device=T.device, dtype=T.dtype)

    use_low = (T < 1000.0007).float().unsqueeze(1)
    a = use_low * a_low + (1.0 - use_low) * a_high
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a.T

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = torch.log(T)

    cp = Rspec * (a1 * invT2 + a2 * invT + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4)

    h = Rspec * T * (
            -a1 * invT2 + a2 * logT * invT + a3 + a4 * T / 2.0 +
            a5 * T2 / 3.0 + a6 * T3 / 4.0 + a7 * T4 / 5.0 + b1 * invT
    )

    s = Rspec * (
            -a1 * invT2 / 2.0 - a2 * invT + a3 * logT +
            a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
    )

    g = Rspec * T * (
            -a1 * invT2 / 2.0 + a2 * (logT + 1.0) * invT +
            a3 * (1.0 - logT) - a4 * T / 2.0 - a5 * T2 / 6.0 -
            a6 * T3 / 12.0 - a7 * T4 / 20.0 + b1 * invT - b2
    )

    p_std = torch.tensor(1e5, device=T.device, dtype=T.dtype)
    s = torch.where(p > 0, s - Rspec * torch.log(p / p_std), s)

    return cp, h, s, g, M