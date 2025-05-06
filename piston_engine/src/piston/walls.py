import numpy as np
from numba import njit


### @njit()
def dqdphi(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi, type1, type2):

    # Woschni heat transfer model
    Pref = ref[0]
    Tref = ref[1]
    Vref = ref[2]
    Psc = ref[3]
    Vsc = ref[4]

    Pbar = P * 1e-5  # convert to bar

    r = 1.0  # c_u/c_m is rotational speed/ mean speed for the gas (tumble)
    v_u = v_mean * r

    if type1 == "HP-process":
        k1 = 1
    elif type1 == "Charge changing":
        k1 = 7
    else:
        print("Unknown type, error in dqdphi")
        return 0.0, 0.0

    dqdphi = 0
    for i, j in zip(Twalls, Awalls):
        if type2:
            if 3.24e-3 > 2.3e-5 * (i - 600) + 5e-3:
                c2 = 3.24e-3
            else:
                c2 = 2.3e-5 * (i - 600) + 5e-3
            # print(gamma)
            Pmotor = Psc * (Vsc / V) ** gamma
            Pmotor_bar = Pmotor * 1e-5
        else:
            c2 = 0
            Pmotor_bar = P

        C = k1 * (2.280 * v_mean + 0.308 * v_u) + c2 * V_d * (Tref / (Pref * Vref)) * (
            Pbar - Pmotor_bar
        )

        alpha = 130.0 * d ** (-0.2) * Pbar**0.8 * C**0.8 * T ** (-0.53)

        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi, alpha


def dqdphi_NASA(T, Twalls, Awalls, type2, dtdphi, D, mu, rho, V):

    # Annand model
    k = 0.070  # heat conductivity air?
    a = 0.35  # 0.35 - 0.8
    b = 0.7  # fixed
    Re = (rho * V * D) / mu

    if type2:  # combustion and expansion
        c = 1.6e-12
    else:  # compression
        c = 0

    dqdphi = 0
    for Tw, A in zip(Twalls, Awalls):
        dqdphi += a * k / D * Re**b * (T - Tw) * A  # + c*(T**4 - Tw**4))* A

    # converting from centrigrade heat unit (CHU) to J? (1.899 Joule)
    dqdphi = dqdphi
    # converting from ft^-2 to m^-2

    return dqdphi * dtdphi


### ### @njit()
def dqdphi_hohenberg(T, P, V, Twalls, Awalls, v_mean, dtdphi):
    # Hohenberg heat loss model
    Pbar = P * 1e-5
    alpha = 130.0 * V ** (-0.06) * Pbar ** (0.8) * T ** (-0.4) * (v_mean + 1.4) ** 0.8
    dqdphi = 0
    for i, j in zip(Twalls, Awalls):
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi


def dqdphi_h2(t, p, twalls, awalls, bore, v_mean, ref, dqfdphi, dtdphi):
    # not good model. not right
    Pref = ref[0]  # pressure at inlet closing
    Tref = ref[1]  # temperature at inlet closing
    Vref = ref[2]  # volume at inlet closing

    pbar = p * 1e-5  # convert to bar

    c1 = 0.08
    c2 = 1.6e-6

    dqfdt = dqfdphi / dtdphi
    alpha = (
        c1
        * bore ** (-0.2)
        * t ** (-0.53)
        * p**0.8
        * (v_mean + c2 * (Tref / (Pref * Vref)) * dqfdt) ** 0.8
    )

    dqdphi = 0
    for i, j in zip(twalls, awalls):
        dqdphi += alpha * j * (t - i) * dtdphi
    return dqdphi


def dqdphi_h2_shudo(
    t, p, V, twalls, awalls, bore, v_mean, ref, dpdphi, dVdphi, gamma, dtdphi
):
    # Modified Woschni-Vogel model from Shudo T Suzuki H 2002
    # https://sci-hub.soik.top/10.1115/icef2002-515
    Pref = ref[0]  # pressure at inlet closing
    Tref = ref[1]  # temperature at inlet closing
    Vref = ref[2]  # volume at inlet closing

    pbar = p * 1e-5  # convert to bar

    c1 = 0.1257
    c2 = 0.8772

    dpdt = dpdphi / dtdphi
    dVdt = dVdphi / dtdphi

    dqdt_apparent = (V * dpdt + gamma * p * dVdt) / (gamma - 1)
    if v_mean + c2 * (Tref / (Pref * Vref)) * dqdt_apparent < 0:
        return 0.0
    alpha = (
        c1
        * bore ** (-0.2)
        * t ** (-0.53)
        * pbar**0.8
        * (v_mean + c2 * (Tref / (Pref * Vref)) * dqdt_apparent) ** 0.8
    )

    dqdphi = 0
    for i, j in zip(twalls, awalls):
        dqdphi += alpha * j * (t - i) * dtdphi
    return dqdphi


### @njit()
def dqdphi_woschni_h2(
    T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi, type1, type2
):
    # Woschni heat transfer model for H2 (the normal one. just dont forget 1.4 ch factor)
    Pref = ref[0]
    Tref = ref[1]
    Vref = ref[2]
    Psc = ref[3]
    Vsc = ref[4]

    Pbar = P * 1e-5

    if type1 == "HP-process":
        # 2.28 original formulation
        c1 = 2.28

        if type2:
            # combustion. that is not during compression
            # 3.24 * 1e-3 original formulation
            # 1.7 * 1e-2 from paper
            # c1 = 10, c2 = 1.7 e-2, for ch = 1 to fit curve. gives 40% heat loss
            c2 = 3.24 * 1e-3
            Pmotor = Psc * (Vsc / V) ** gamma
            Pmotor_bar = Pmotor * 1e-5

        else:
            # compression phase
            c2 = 0
            Pmotor_bar = P
            Pmotor = P

    elif type1 == "Charge changing":
        # 6.18 original formulation
        c1 = 6.18
        c2 = 0
        Pmotor_bar = P
        Pmotor = P
    else:
        print("Unknown type, error in dqdphi")
        return 0.0, 0.0

    dqdphi = 0
    # alpha = 0
    for i, j in zip(Twalls, Awalls):

        C = c1 * v_mean + c2 * V_d * (Tref / (Pref * Vref)) * (P - Pmotor)
        alpha = 0.012991 * d ** (-0.2) * P**0.8 * C**0.8 * T ** (-0.53)
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi, alpha
