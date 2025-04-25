from thermo import mixture


def stagnation(ps, ts, m):
    """Returns stagnation temperature and pressure, given static temperature, static pressure and mach number"""
    R_uni = 8.3144626  # J mol^-1 K^-1
    # Assuming pure air
    _, _, cp, _, _, _, _, M = mixture(ts, ps, equ=0)
    R = R_uni / M
    cv = cp - R
    gamma = cp / cv

    t = ts * (1 + (gamma - 1) * 0.5 * m**2)
    p = ps * (1 + (gamma - 1) * 0.5 * m**2) ** (gamma / (gamma - 1))

    return p, t
