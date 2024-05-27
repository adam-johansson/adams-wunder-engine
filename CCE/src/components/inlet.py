def inlet(p1, t1, dp):
    """Returns total pressure and temperature after intake. Input: total pressure and temperature before intake
    and pressure loss"""

    p2 = p1 * (1 - dp)
    # Adiabatic intake
    t2 = t1

    return p2, t2
