from math import log
from numba import njit

# Universal gas constant from NASA polynomials pdf
R = 8.314510  # J mol^-1 K^-1

# Also from NASA pdf
# Sackur-Tetrode constant S^0/R for p0 = 1 bar is -1.151693


# INFORMATION:
# Inside the polynomials, the units are kJ/mol


@njit()
def N2(T, p):
    T = min(6000.0, max(200.0, T))  # clamp temperature
    M = 28.0134e-3
    Rspec = R / M

    # Coefficient arrays
    a_low = [
        2.210371497e04, -3.818461820e02, 6.082738360e00,
        -8.530914410e-03, 1.384646189e-05, -9.625793620e-09, 2.519705809e-12,
        7.108460860e02, -1.076003744e01,
    ]
    a_high = [
        5.877124060e05, -2.239249073e03, 6.066949220e00,
        -6.139685500e-04, 1.491806679e-07, -1.923105485e-11, 1.061954386e-15,
        1.283210415e04, -1.586640027e01,
    ]

    # Choose coefficients without branching
    use_low = 1.0 if T < 1000.0007 else 0.0  # 1.0 if low, 0.0 if high
    a = [use_low * al + (1.0 - use_low) * ah for al, ah in zip(a_low, a_high)]
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = a

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = log(T)

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

    p_std = 1e5
    if p > 0:
        s -= Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def O2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    T = min(6000.0, max(200.0, T))  # clamp temperature
    M = 31.9988e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = -3.425563420e04
        a2 = 4.847000970e02
        a3 = 1.119010961e00
        a4 = 4.293889240e-03
        a5 = -6.836300520e-07
        a6 = -2.023372700e-09
        a7 = 1.039040018e-12
        b1 = -3.391454870e03
        b2 = 1.849699470e01
    else:  # 1000K to float(6000)K
        a1 = -1.037939022e06
        a2 = 2.344830282e03
        a3 = 1.819732036e00
        a4 = 1.267847582e-03
        a5 = -2.188067988e-07
        a6 = 2.053719572e-11
        a7 = -8.193467050e-16
        b1 = -1.689010929e04
        b2 = 1.738716506e01


    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = log(T)

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

    p_std = 1e5
    if p > 0:
        s -= Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def Ar(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    M = 39.948e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    T = min(6000.0, max(200.0, T))  # clamp temperature

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 0.0
        a2 = 0.0
        a3 = 2.500000000e00
        a4 = 0.0
        a5 = 0.0
        a6 = 0.0
        a7 = 0.0
        b1 = -7.453750000e02
        b2 = 4.379674910e00
    else:  # 1000K to float(6000)K
        a1 = 2.010538475e01
        a2 = -5.992661070e-02
        a3 = 2.500069401e00
        a4 = -3.992141160e-08
        a5 = 1.205272140e-11
        a6 = -1.819015576e-15
        a7 = 1.078576636e-19
        b1 = -7.449939610e02
        b2 = 4.379180110e00

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = log(T)

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

    p_std = 1e5
    if p > 0:
        s -= Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def CO2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford

    # 9365.469 H(295.15) - H(0) J/mol
    M = 44.0095e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    T = min(6000.0, max(200.0, T))  # clamp temperature
    if T < 1000.0007:  # between 200K and 1000K
        a1 = 4.943650540e04
        a2 = -6.264116010e02
        a3 = 5.301725240e00
        a4 = 2.503813816e-03
        a5 = -2.127308728e-07
        a6 = -7.689988780e-10
        a7 = 2.849677801e-13
        b1 = -4.528198460e04
        b2 = -7.048279440e00
    else:  # 1000K to float(6000)K
        a1 = 1.176962419e05
        a2 = -1.788791477e03
        a3 = 8.291523190e00
        a4 = -9.223156780e-05
        a5 = 4.863676880e-09
        a6 = -1.891053312e-12
        a7 = 6.330036590e-16
        b1 = -3.908350590e04
        b2 = -2.652669281e01

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = log(T)

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

    p_std = 1e5
    if p > 0:
        s -= Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def H2O(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    M = 18.01528e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    T = min(6000.0, max(200.0, T))  # clamp temperature
    if T < 1000.0007:  # between 200K and 1000K
        a1 = -3.947960830e04
        a2 = 5.755731020e02
        a3 = 9.317826530e-01
        a4 = 7.222712860e-03
        a5 = -7.342557370e-06
        a6 = 4.955043490e-09
        a7 = -1.336933246e-12
        b1 = -3.303974310e04
        b2 = 1.724205775e01
    else:  # 1000K to float(6000)K
        a1 = 1.034972096e06
        a2 = -2.412698562e03
        a3 = 4.646110780e00
        a4 = 2.291998307e-03
        a5 = -6.836830480e-07
        a6 = 9.426468930e-11
        a7 = -4.822380530e-15
        b1 = -1.384286509e04
        b2 = -7.978148510e00

    # Precompute powers and logs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    invT = 1.0 / T
    invT2 = invT * invT
    logT = log(T)

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

    p_std = 1e5
    if p > 0:
        s -= Rspec * log(p / p_std)

    return cp, h, s, g, M

@njit()
def JETA_L(T):
    # LIQUID JET A

    if T < 220 or T > 550:
        print(f"Temperature {T} outside valid range for Jet-A.")
        return
    M = 167.31102e-3  # kg/mol
    Rspec = R / M

    a1 = -4.218262130e05
    a2 = -5.576600450e03
    a3 = 1.522120958e02
    a4 = -8.610197550e-01
    a5 = 3.071662234e-03
    a6 = -4.702789540e-06
    a7 = 2.743019833e-09
    b1 = -3.238369150e04
    b2 = -6.781094910e02

    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )  # J kg^-1 K^-1
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )

    # enthalpy of formation DONT USE THIS
    #hf = -303403.000 / M
    #hf = 0
    #h = h - hf  # J kg^-1

    return cp, h, s, M


@njit()
def JETA_G(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    # Gaseous Jet A

    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 273.150:
        T = 273.150
        # print(f'Temperature under 200 K was found')

    M = 167.31102e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = -6.068695590e05
        a2 = 8.328259590e03
        a3 = -4.312321270e01
        a4 = 2.572390455e-01
        a5 = -2.629316040e-04
        a6 = 1.644988940e-07
        a7 = -4.645335140e-11
        b1 = -7.606962760e04
        b2 = 2.794305937e02
    else:  # 1000K to float(6000)K
        a1 = 1.858356102e07
        a2 = -7.677219890e04
        a3 = 1.419826133e02
        a4 = -7.437524530e-03
        a5 = 5.856202550e-07
        a6 = 1.223955647e-11
        a7 = -3.149201922e-15
        b1 = 4.221989520e05
        b2 = -8.986061040e02

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    # pressure dependence of the Gibbs
    # g = g + T * Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def H2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    T = min(6000.0, max(200.0, T))  # clamp temperature
    M = 2.0158800e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 4.078323210e04
        a2 = -8.009186040e02
        a3 = 8.214702010e00
        a4 = -1.269714457e-02
        a5 = 1.753605076e-05
        a6 = -1.202860270e-08
        a7 = 3.368093490e-12
        b1 = 2.682484665e03
        b2 = -3.043788844e01
    else:  # 1000K to float(6000)K
        a1 = 5.608128010e05
        a2 = -8.371504740e02
        a3 = 2.975364532e00
        a4 = 1.252249124e-03
        a5 = -3.740716190e-07
        a6 = 5.936625200e-11
        a7 = -3.606994100e-15
        b1 = 5.339824410e03
        b2 = -2.202774769e00

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    # pressure dependence of the Gibbs
    # g = g + T * Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def CO(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 28.0101000e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 1.489045326e04
        a2 = -2.922285939e02
        a3 = 5.724527170e00
        a4 = -8.176235030e-03
        a5 = 1.456903469e-05
        a6 = -1.087746302e-08
        a7 = 3.027941827e-12
        b1 = -1.303131878e04
        b2 = -7.859241350e00
    else:  # 1000K to float(6000)K
        a1 = 4.619197250e05
        a2 = -1.944704863e03
        a3 = 5.916714180e00
        a4 = -5.664282830e-04
        a5 = 1.398814540e-07
        a6 = -1.787680361e-11
        a7 = 9.620935570e-16
        b1 = -2.466261084e03
        b2 = -1.387413108e01

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = -110535.196 / M
    hf = 0
    h = h - hf

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def H(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 1.0079400e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 0.00
        a2 = 0.00
        a3 = 2.500000000e00
        a4 = 0.00
        a5 = 0.00
        a6 = 0.00
        a7 = 0.00
        b1 = 2.547370801e04
        b2 = -4.466828530e-01
    else:  # 1000K to float(6000)K
        a1 = 6.078774250e01
        a2 = -1.819354417e-01
        a3 = 2.500211817e00
        a4 = -1.226512864e-07
        a5 = 3.732876330e-11
        a6 = -5.687744560e-15
        a7 = 3.410210197e-19
        b1 = 2.547486398e04
        b2 = -4.481917770e-01

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = +217998.828 / M
    hf = 0
    h = h - hf  # (J/kg)

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def N(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 14.0067000e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 0.00
        a2 = 0.00
        a3 = 2.500000000e00
        a4 = 0.00
        a5 = 0.00
        a6 = 0.00
        a7 = 0.00
        b1 = 5.610463780e04
        b2 = 4.193905036e00
    else:  # 1000K to float(6000)K
        a1 = 8.876501380e04
        a2 = -1.071231500e02
        a3 = 2.362188287e00
        a4 = 2.916720081e-04
        a5 = -1.729515100e-07
        a6 = 4.012657880e-11
        a7 = -2.677227571e-15
        b1 = 5.697351330e04
        b2 = 4.865231506e00

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = +472680.000 / M
    hf = 0
    h = h - hf  # (J/kg)

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def NO(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 30.0061000e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = -1.143916503e04
        a2 = 1.536467592e02
        a3 = 3.431468730e00
        a4 = -2.668592368e-03
        a5 = 8.481399120e-06
        a6 = -7.685111050e-09
        a7 = 2.386797655e-12
        b1 = 9.098214410e03
        b2 = 6.728725490e00
    else:  # 1000K to float(6000)K
        a1 = 2.239018716e05
        a2 = -1.289651623e03
        a3 = 5.433936030e00
        a4 = -3.656034900e-04
        a5 = 9.880966450e-08
        a6 = -1.416076856e-11
        a7 = 9.380184620e-16
        b1 = 1.750317656e04
        b2 = -8.501669090e00

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = +91271.310 / M
    hf = 0
    h = h - hf  # (J/kg)

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def O(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 15.9994000e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = -7.953611300e03
        a2 = 1.607177787e02
        a3 = 1.966226438e00
        a4 = 1.013670310e-03
        a5 = -1.110415423e-06
        a6 = 6.517507500e-10
        a7 = -1.584779251e-13
        b1 = 2.840362437e04
        b2 = 8.404241820e00
    else:  # 1000K to float(6000)K
        a1 = 2.619020262e05
        a2 = -7.298722030e02
        a3 = 3.317177270e00
        a4 = -4.281334360e-04
        a5 = 1.036104594e-07
        a6 = -9.438304330e-12
        a7 = 2.725038297e-16
        b1 = 3.392428060e04
        b2 = -6.679585350e-01

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = +249175.003 / M
    hf = 0
    h = h - hf  # (J/kg)

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M


@njit()
def OH(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > float(6000):
        T = float(6000)
        # print(f'Temperature over float(6000) K was found')
    if T < 200:
        T = float(200)
        # print(f'Temperature under 200 K was found')
    M = 17.0073400e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = -1.998858990e03
        a2 = 9.300136160e01
        a3 = 3.050854229e00
        a4 = 1.529529288e-03
        a5 = -3.157890998e-06
        a6 = 3.315446180e-09
        a7 = -1.138762683e-12
        b1 = 2.991214235e03
        b2 = 4.674110790e00
    else:  # 1000K to float(6000)K
        a1 = 1.017393379e06
        a2 = -2.509957276e03
        a3 = 5.116547860e00
        a4 = 1.305299930e-04
        a5 = -8.284322260e-08
        a6 = 2.006475941e-11
        a7 = -1.556993656e-15
        b1 = 2.019640206e04
        b2 = -1.101282337e01

    # mass specific constant pressure heat capacity
    cp = Rspec * (
        a1 * T ** (-2.0)
        + a2 * T ** (-1.0)
        + a3
        + a4 * T
        + a5 * T ** 2.0
        + a6 * T ** 3.0
        + a7 * T ** 4.0
    )
    # mass specific enthalpy
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0)
            + a2 * log(T) / T
            + a3
            + a4 * T / 2.0
            + a5 * T ** 2.0 / 3.0
            + a6 * T ** 3.0 / 4.0
            + a7 * T ** 4.0 / 5.0
            + b1 / T
        )
    )
    # standard state entropy, per mass
    s = Rspec * (
        -a1 * T ** (-2.0) / 2
        - a2 * T ** (-1.0)
        + a3 * log(T)
        + a4 * T
        + a5 * T ** 2.0 / 2
        + a6 * T ** 3.0 / 3.0
        + a7 * T ** 4.0 / 4.0
        + b2
    )
    # standard state molar Gibbs free energy, per mass
    g = (
        Rspec
        * T
        * (
            -a1 * T ** (-2.0) / 2
            + a2 * T ** (-1.0) * (log(T) + 1)
            + a3 * (1 - log(T))
            - a4 * T / 2.0
            - a5 * T ** 2.0 / 6.0
            - a6 * T ** 3.0 / 12.0
            - a7 * T ** 4.0 / 20.0
            + b1 / T
            - b2
        )
    )

    # add enthalpy of formation or something like that (heat of formation)
    # divide by M to make it per kg instead of per mole
    hf = +37278.206 / M
    hf = 0
    h = h - hf  # (J/kg)

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    if p > 0:
        s = s - Rspec * log(p / p_std)

    return cp, h, s, g, M
