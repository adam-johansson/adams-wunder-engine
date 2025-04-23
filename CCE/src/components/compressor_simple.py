def compressor_simple(p, t, eta_c, pi_c, gamma, cp, type):
    if type == "Isen":
        t_out = t + (t / eta_c) * (pi_c ** ((gamma - 1) / gamma) - 1)
    elif type == "Poly":
        # eta_c is polytropic efficency
        t_out = t * (pi_c) ** ((gamma - 1) / (gamma * eta_c))
    else:
        raise Exception("unknown compression type")

    p_out = p * pi_c
    w = cp * (t_out - t)

    return p_out, t_out, w
