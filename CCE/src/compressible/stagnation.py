from thermo import mixture


def stagnation(ps, ts, m):
    """Returns stagnation temperature and pressure, given static temperature, static pressure and mach number"""

    _, _, _, _, _, gamma, _, _ = mixture(ts, ps, equ=0)

    t = ts * (1 + (gamma - 1) * 0.5 * m**2)
    p = ps * (1 + (gamma - 1) * 0.5 * m**2) ** (gamma / (gamma - 1))

    return p, t
