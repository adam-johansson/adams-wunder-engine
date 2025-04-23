import numpy as np


def thrust(Fh_s, Fc_s, ca):

    Fs = Fh_s + Fc_s - ca

    return Fs


def mass_flow(ca, pa, Ta, far, BPR, p8, rho8, c8, p18, rho18, c18, Fn):

    t1 = (1 + far) / (1 + BPR) * (c8 + (p8 - pa) / (rho8 * c8))
    t2 = BPR / (1 + BPR) * (c18 + (p18 - pa) / (rho18 * c18))
    mdot_0 = Fn / (t1 + t2 - ca)
    mdot_h = mdot_0 * (1 + far) / (1 + BPR)
    mdot_c = mdot_0 * BPR / (1 + BPR)
    mdot_f = mdot_0 * far / (1 + BPR)

    return mdot_0, mdot_h, mdot_c, mdot_f
