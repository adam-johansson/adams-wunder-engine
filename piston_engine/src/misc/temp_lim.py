def t_in_lim(p_in):

    pr = p_in / 0.277e5
    eta = 0.85
    T1 = 238

    # piston inlet temperature and pressure
    T2 = T1 * (pr) ** ((1.4 - 1) / (1.4 * eta))

    return T2
