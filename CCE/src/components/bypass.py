from CCE.src.components import hx, mix

def bypass(p1, t1, dp, m1, heat_power):
    # just to fit
    hx_frac = 0.0652  # interpolate from TO example
    m_hx = m1 * hx_frac  # kg/s
    dp_hx = 5.7 / 100  # pressure loss over hx

    # pressure loss in bypass duct
    p15 = p1 * (1 - dp)
    # adiabatic bypass duct
    t15 = t1

    # heat exchanger
    p2, t2 = hx(p15, t15, dp_hx, heat_power, m_hx)

    p25 = (p15 * (m1 - m_hx) + p2 * m_hx) / m1

    # mix after heat exchanger
    t25, equ25 = mix(m1 - m_hx, t15, 0, m_hx, t2, 0)

    return p25, t25
