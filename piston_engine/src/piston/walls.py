import numpy as np
from numba import njit



@njit()
def dqdphi_woschni(
    T: float,
    p: float,
    V: float,
    gamma: float,
    Twalls: list,
    Awalls: list,
    v_mean: float,
    d: float,
    V_d: float,
    ref: list,
    dtdphi: float,
    hp_phase: bool,
    firing: bool,
) -> tuple:
    """
    Computes heat loss dQ/dphi using the Woschni model adapted for hydrogen combustion.

    Parameters:
        T        : in-cylinder gas temperature [K]
        p        : in-cylinder pressure [Pa]
        V        : instantaneous volume [m^3]
        gamma    : specific heat ratio
        Twalls   : list of wall temperatures [K]
        Awalls   : list of corresponding wall surface areas [m^2]
        v_mean   : mean piston speed [m/s]
        d        : cylinder bore [m]
        V_d      : displaced volume [m^3]
        ref      : reference values [Pref, Tref, Vref, Psc, Vsc]
        dtdphi   : derivative dt/dphi [s/rad]
        type1    : process type, "HP-process" or "Charge changing"
        type2    : combustion phase indicator (True if burning)

    Returns:
        dqdphi : total heat loss rate [W/rad]
        alpha  : representative heat transfer coefficient [W/m²·K]
    """
    # Woschni heat transfer model (for H2 just dont forget 1.4 ch factor)
    Pref = ref[0]
    Tref = ref[1]
    Vref = ref[2]
    Psc = ref[3]
    Vsc = ref[4]

    p_bar = p * 1e-5

    # intake swirl (range 0 to 3 according to textbook. I use middle value)
    swirl = 1.5

    # during high pressure phase (valves closed)
    if hp_phase:
        c1 = 2.28 + 0.308 * swirl

        if firing:
            # combustion. that is not during compression
            c2 = 3.24*1e-3
            Pmotor = Psc * (Vsc / V) ** gamma

        else:
            # compression phase
            c2 = 0
            Pmotor = p

    # charge changing phase (valves open)
    else:
        c1 = 6.18 + 0.417 * swirl
        c2 = 0
        Pmotor = p

    dqdphi = 0

    # estimation of gas velocity
    w_gas = c1 * v_mean + c2 * V_d * (Tref / (Pref * Vref)) * (p - Pmotor)

    # heat transfer coefficient
    alpha = 127.93 * d ** (-0.2) * p_bar ** 0.8 * w_gas ** 0.8 * T ** (-0.53)

    for i, j in zip(Twalls, Awalls):
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi, alpha

@njit()
def dqdphi_woschni_h2(
    T: float,
    p: float,
    V: float,
    gamma: float,
    Twalls: list,
    Awalls: list,
    v_mean: float,
    d: float,
    V_d: float,
    ref: list,
    dtdphi: float,
    hp_phase: bool,
    firing: bool,
) -> tuple:
    """
    Computes heat loss dQ/dphi using the Woschni model adapted for hydrogen combustion.

    Parameters:
        T        : in-cylinder gas temperature [K]
        p        : in-cylinder pressure [Pa]
        V        : instantaneous volume [m^3]
        gamma    : specific heat ratio
        Twalls   : list of wall temperatures [K]
        Awalls   : list of corresponding wall surface areas [m^2]
        v_mean   : mean piston speed [m/s]
        d        : cylinder bore [m]
        V_d      : displaced volume [m^3]
        ref      : reference values [Pref, Tref, Vref, Psc, Vsc]
        dtdphi   : derivative dt/dphi [s/rad]
        type1    : process type, "HP-process" or "Charge changing"
        type2    : combustion phase indicator (True if burning)

    Returns:
        dqdphi : total heat loss rate [W/rad]
        alpha  : representative heat transfer coefficient [W/m²·K]
    """
    # Woschni heat transfer model (for H2 just dont forget 1.4 ch factor)
    Pref = ref[0]
    Tref = ref[1]
    Vref = ref[2]
    Psc = ref[3]
    Vsc = ref[4]

    p_bar = p * 1e-5

    # intake swirl (range 0 to 3 according to textbook. I use middle value)
    swirl = 1.5

    # during high pressure phase (valves closed)
    if hp_phase:
        c1 = 2.28 + 0.308 * swirl

        if firing:
            # combustion. that is not during compression
            c2 = 3.24*1e-3
            Pmotor = Psc * (Vsc / V) ** gamma

        else:
            # compression phase
            c2 = 0
            Pmotor = p

    # charge changing phase (valves open)
    else:
        c1 = 6.18 + 0.417 * swirl
        c2 = 0
        Pmotor = p

    dqdphi = 0

    # estimation of gas velocity
    w_gas = c1 * v_mean + c2 * V_d * (Tref / (Pref * Vref)) * (p - Pmotor)

    # double gas velocity because hydrogen is more turbulent
    w_gas = w_gas * 1

    # heat transfer coefficient
    alpha = 127.93 * d ** (-0.2) * p_bar ** 0.8 * w_gas ** 0.8 * T ** (-0.53)

    for i, j in zip(Twalls, Awalls):
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi, alpha


@njit()
def dqdphi_hohenberg(T, P, V, Twalls, Awalls, v_mean, dtdphi):

    # Hohenberg heat loss model (this is the orignal fomrulation from his paper OG paper)
    Pbar = P * 1e-5
    alpha = 130.0 * V ** (-0.06) * Pbar ** (0.8) * T ** (-0.4) * (v_mean + 1.4) ** 0.8
    dqdphi = 0
    for i, j in zip(Twalls, Awalls):
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi

@njit()
def dqdphi_michl(D, T, p, R_spec, V, Twalls, Awalls, dtdphi):

    # Michl heat loss model from the paper "Derivation and validation of a heat transfer model in a hydrogen combustion engine"
    # THIS IS FOR HYDROGEN

    # Characteristic length
    # D is bore
    L_c = D ** -0.23

    # density
    rho = p / (T * R_spec)

    # thermal conductivity
    kappa_air = 2.86 * 1e-4 * T**0.785
    kappa_h2 = 2.32 * 1e-3 * T**0.760
    kappa_burned = 7.36 * 1e-5 ** T**1.031

    # dynamic viscosity
    eta_air = 4.39 * 1e-7 * T**0.662
    eta_h2 = 2.27 * 1e-7 * T**0.646
    eta_burned = 1.80 * 1e-7 * T**0.779

    # Fluid property term
    Omega = kappa * (rho / eta)**0.77

    dqdphi = 0
    for i, j in zip(Twalls, Awalls):
        alpha = C * L_c * Omega * v_c * Xi
        dqdphi += alpha * j * (T - i) * dtdphi
    return dqdphi



@njit()
def dqdphi_moreadvanced(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi, type1, type2):

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
