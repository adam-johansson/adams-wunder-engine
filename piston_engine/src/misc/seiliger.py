import math


def seiliger(p1, t1, cr, far, bore, fuel):

    # top dead center pressure estimate
    p_tdc = p1 * (cr) ** 1.4 * 0.9
    t_tdc = t1 * cr**0.4

    # isochoric heat addition (assuming constant mass) (neglecting change in specific gas constant)
    cp = 1004
    cv = cp / 1.4

    # lower heating value h2
    if fuel == "h2":
        lhv = 119.96e6
    else:
        lhv = 42.8e6

    # cylinder volume
    V = bore * (bore / 2) ** 2 * math.pi

    # air density
    R_air = 287
    rho_air = p1 / (R_air * t1)

    # air in cylinder
    m_air = rho_air * V

    # mass of fuel injected
    m_h2 = m_air * far

    # amount of heat added (factor 0.6 from Kaiser paper)
    Q = lhv * m_h2 * 0.6

    # assuming total mas is air
    t_max = t_tdc + Q / (m_air * cv)

    # assumptions of onl 40% of max pressure (from Unified Thermodynamic... Kaiser)
    p_max = p_tdc * (t_max / t_tdc) * 0.4 + 0.6 * p_tdc

    return p_max
