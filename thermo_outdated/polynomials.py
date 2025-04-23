import numpy as np

# Universal gas constant (from Wikipedia) or maybe the database?
R = 8.3144626  # J mol^-1 K^-1


def N2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford

    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')

    M = 28.0134e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1

    if T < 1000.0007:  # between 200K and 1000K
        a1 = 2.210371497e04
        a2 = -3.818461820e02
        a3 = 6.082738360e00
        a4 = -8.530914410e-03
        a5 = 1.384646189e-05
        a6 = -9.625793620e-09
        a7 = 2.519705809e-12
        b1 = 7.108460860e02
        b2 = -1.076003744e01
    else:  # 1000K to 6000K
        a1 = 5.877124060e05
        a2 = -2.239249073e03
        a3 = 6.066949220e00
        a4 = -6.139685500e-04
        a5 = 1.491806679e-07
        a6 = -1.923105485e-11
        a7 = 1.061954386e-15
        b1 = 1.283210415e04
        b2 = -1.586640027e01

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )

    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )

    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )

    # reference 0K
    # h = h + 8.670104*1e3 / M

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    s = s - Rspec * np.log(p / p_std)

    return cp, h, s, M


def O2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')
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
    else:  # 1000K to 6000K
        a1 = -1.037939022e06
        a2 = 2.344830282e03
        a3 = 1.819732036e00
        a4 = 1.267847582e-03
        a5 = -2.188067988e-07
        a6 = 2.053719572e-11
        a7 = -8.193467050e-16
        b1 = -1.689010929e04
        b2 = 1.738716506e01

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )
    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )

    # reference 0K
    h = h + 8.680104e3 / M

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    s = s - Rspec * np.log(p / p_std)

    return cp, h, s, M


def Ar(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    M = 39.948e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')
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
    else:  # 1000K to 6000K
        a1 = 2.010538475e01
        a2 = -5.992661070e-02
        a3 = 2.500069401e00
        a4 = -3.992141160e-08
        a5 = 1.205272140e-11
        a6 = -1.819015576e-15
        a7 = 1.078576636e-19
        b1 = -7.449939610e02
        b2 = 4.379180110e00

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )
    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )

    # reference 0K
    h = h + 6.197 * 1e3 / M

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    s = s - Rspec * np.log(p / p_std)

    return cp, h, s, M


def CO2(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford

    # 9365.469 H(295.15) - H(0) J/mol
    M = 44.0095e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')
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
    else:  # 1000K to 6000K
        a1 = 1.176962419e05
        a2 = -1.788791477e03
        a3 = 8.291523190e00
        a4 = -9.223156780e-05
        a5 = 4.863676880e-09
        a6 = -1.891053312e-12
        a7 = 6.330036590e-16
        b1 = -3.908350590e04
        b2 = -2.652669281e01

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )

    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )

    # WORKING IS TO USE THESE FOR CO2 and H2O and none for N2, Ar and O2
    # enthalpy of formation 298.15 K
    # h = h + 393510.000 / M

    # enthalpy of formation 0k
    # h = h + 393142.000 / M

    # standard enthalpy 0K
    h = h + 402875.0 / M

    # h = h + 9365.469 / M

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    s = s - Rspec * np.log(p / p_std)

    return cp, h, s, M


def H2O(T, p):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    M = 18.01528e-3  # kg/mol
    Rspec = R / M  # J kg^-1 K^-1
    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')
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
    else:  # 1000K to 6000K
        a1 = 1.034972096e06
        a2 = -2.412698562e03
        a3 = 4.646110780e00
        a4 = 2.291998307e-03
        a5 = -6.836830480e-07
        a6 = 9.426468930e-11
        a7 = -4.822380530e-15
        b1 = -1.384286509e04
        b2 = -7.978148510e00

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )  # J kg^-1 K^-1

    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )

    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )

    # 298.15 k
    # h = h + 241826.000 / M    #J kg^-1

    # enthalpy of formation
    # h = h + 238922.0 / M

    # enthalpy at 0K
    h = h + 251730.0 / M

    # h = h + 9.904 / M

    # h = h +

    # standards state pressure is 1 bar
    p_std = 1e5

    # pressure dependence of the entropy
    s = s - Rspec * np.log(p / p_std)

    #
    # s = s - 3500

    return cp, h, s, M


def OCTAN(T):
    M = 114.2285200e-3
    Rspec = R / M

    a1 = -6.986647150e05
    a2 = 1.338501096e04
    a3 = -8.415165920e01
    a4 = 3.271936660e-01
    a5 = -3.777209590e-04
    a6 = 2.339836988e-07
    a7 = -6.010892650e-11
    b1 = -9.026223250e04

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )  # J kg^-1 K^-1
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )
    h = h + 208750.000 / M  # J kg^-1
    return cp, h


def JETA(T):
    if T < 220 or T > 550:
        print(f"Temperature {T} outside valid range for Jet-A.")
        return
    M = 167.31102e-3
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
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )  # J kg^-1 K^-1
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )
    h = h + 303.403e3 / M  # - 10.340270190732554 #J kg^-1
    return cp, h


def H2(T):
    # This is NASA 9 polynomial from NASA Glenn Coefficients for Calculating
    # Thermodynamic Properties of Individual Species 2002 Bonnie, McBride and Sanford
    if T > 6000:
        T = 6000
        # print(f'Temperature over 6000 K was found')
    if T < 200:
        T = 200
        # print(f'Temperature under 200 K was found')
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
    else:  # 1000K to 6000K
        a1 = 5.608128010e05
        a2 = -8.371504740e02
        a3 = 2.975364532e00
        a4 = 1.252249124e-03
        a5 = -3.740716190e-07
        a6 = 5.936625200e-11
        a7 = -3.606994100e-15
        b1 = 5.339824410e03
        b2 = -2.202774769e00

    cp = Rspec * (
        a1 * T ** (-2)
        + a2 * T ** (-1)
        + a3
        + a4 * T
        + a5 * T**2
        + a6 * T**3
        + a7 * T**4
    )
    h = (
        Rspec
        * T
        * (
            -a1 * T ** (-2)
            + a2 * np.log(T) / T
            + a3
            + a4 * T / 2
            + a5 * T**2 / 3
            + a6 * T**3 / 4
            + a7 * T**4 / 5
            + b1 / T
        )
    )
    s = Rspec * (
        -a1 * T ** (-2) / 2
        - a2 * T ** (-1)
        + a3 * np.log(T)
        + a4 * T
        + a5 * T**2 / 2
        + a6 * T**3 / 3
        + a7 * T**4 / 4
        + b2
    )
    return cp, h, s, M
