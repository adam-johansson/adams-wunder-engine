from CCE.src import thermo_outdated

def stagnation(ps, ts, m):
    """Returns stagnation temperature and pressure, given static temperature, static pressure and mach number"""
    R_uni = 8.3144626  # J mol^-1 K^-1
    # Assuming pure air
    cp, h, s, Molar = thermo_outdated.properties(ts, ps, equ=0)
    R = R_uni / Molar
    cv = cp - R
    gamma = cp / cv

    t = ts * (1 + (gamma - 1) * 0.5 * m ** 2)
    p = ps * (1 + (gamma - 1) * 0.5 * m ** 2) ** (gamma / (gamma - 1))

    return p, t

