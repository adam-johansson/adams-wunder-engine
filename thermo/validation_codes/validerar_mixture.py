from thermo import mixture






h, u, cp, cv, R, gamma, s, M = mixture(T, p, equ, fuel_type="H2", premixed=True)