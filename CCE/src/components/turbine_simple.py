# remake this function soon


def turbine_simple(
    p1, T1, cp, gamma, w, eta, far, bpr_c, type, cooling, alpha=0, T_cooling=0, cp_c=0
):
    if not cooling:
        ws = w / ((1 + far) * (1 - bpr_c) + bpr_c)
        T2 = T1 - ws / cp  # temperature reached after performing necessary work

        if type == "Isen":
            pr = (1 - (1 / eta) * (1 - T2 / T1)) ** (gamma / (gamma - 1))
            p2 = p1 * pr

        elif type == "Poly":
            p2 = p1 * (T1 / T2) ** (-gamma / (eta * (gamma - 1)))

        else:
            print("Unknown expansion type.")
    elif cooling:

        g_1 = (1 - bpr_c) * (1 + far)
        g_cools = bpr_c * alpha
        g_b = g_1 + g_cools
        g_coolr = bpr_c * (1 - alpha)
        g_2 = g_1 + bpr_c

        T_b = (g_1 * cp * T1 + g_cools * cp_c * T_cooling) / (
            g_b * cp
        )  # Temperature after stator, before rotor

        w_s = w / g_b
        T_c = (
            T_b - w_s / cp
        )  # assuming cp of hot exhaust gas temperature after rotor (before mixing with cooling air)
        p2 = p1 * (T_b / T_c) ** (-gamma / (eta * (gamma - 1)))

        # mix again
        T2 = (g_b * cp * T_c + g_coolr * cp_c * T_cooling) / (
            g_2 * cp
        )  # Temperature after mixing after rotor

    else:
        print("Wrong cooling.")

    return p2, T2
