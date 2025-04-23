from CoolProp.CoolProp import PropsSI

from scipy.optimize import fsolve


def fuel_oil_hx(p1, t1, dp, t2, m1):

    # ----------------------------------Fluid inputs----------------------------------

    # assuming pressure losses
    dp_h = 0.06
    dp_c = 0.057

    # hot oil
    th_i = 500
    ph_i = 10e5
    m_h = 6.6
    cp_h = PropsSI("Cpmass", "T", th_i, "P", ph_i, "INCOMP::PNF2")

    # cold hydrogen
    tc_i = 20
    pc_i = 200e5
    m_c = 0.080
    hc_i = PropsSI("Hmass", "T", tc_i, "P", pc_i, "Hydrogen")

    # assuming constant cp for the oil
    C_oil = m_h * cp_h

    # assuming 90% efficiency
    eps = 0.90

    # assuming 5% pressure loss
    # ph_o = ph_i * (1 - dp_h)
    pc_o = pc_i * (1 - dp_c)

    def find_tc_o(tc_o):
        hc_o = PropsSI("Hmass", "T", tc_o, "P", pc_o, "Hydrogen")

        cp_c = (hc_o - hc_i) / (tc_o - tc_i)

        Cmin = cp_c * m_c

        q = eps * Cmin * (th_i - tc_i)

        tc_o2 = tc_i + q / Cmin

        return tc_o - tc_o2

    tc_o = fsolve(find_tc_o, x0=460)[0]

    hc_o = PropsSI("Hmass", "T", tc_o, "P", pc_o, "Hydrogen")

    Cmin = (hc_o - hc_i) / (tc_o - tc_i) * m_c

    q = eps * Cmin * (th_i - tc_i)

    th_o = th_i - q / (C_oil)

    # returning total heat rejected and outlet oil temp
    return q, th_o
