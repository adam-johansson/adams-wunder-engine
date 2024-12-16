



def equilibrium_OHC(t, equ, p):
    """
    Function that calculates the chemical equilibrium of oxygen, hydrogen and carbon (OHC). This is the zone 2,
    behind the flame front, of hydrocarbon combustion.

    These reactions proceed much faster than the reactions of the Zeldovich mechanism, (at the temperature of the
    cylinder), so we assume that we always have partial equilibrium.

    We have 5 chemical reaction equations:

    (1) H2 = 2 H
    (2) O2 = 2 O
    (3) H2O = (1/2) H2 + OH
    (4) H2O = (1/2) O2 + H2
    (5) CO2 = CO + (1/2) O2

    Further we have mass conservation of the atoms O, H and C.

    Combined with partial pressures must add up to the total pressure of the system.



    :param t:
    :param equ:
    :param p:
    :return:
    """
    # THIS IS FOR GRI_MECH, Unimolecular and recombination reactions
    # k = A T^m exp(-E/RT)
    # conentrations in mol/cm^3
    # A = [1/s, cm^3/mol/s,cm^6/mol^2/s] first, second and third order reactions respectively
    # T is in Kelvin
    # E is [cal/mol]


    # KOEFFICIENTS                              A            n            E_A
    # 2H+M<=>H2+M                              1.000E+18   -1.000        .00

    # Equilibrium constant. Ratio between forward and reverse reactions
    #K_c = k_f / k_r

    # partial equilibrium
    # [H]^2/[H_2] = K_C4
    # [O]^2 / [O_2] = K_C5, osv

    # Behöver Kc för alla reaktioner. Därför behöver jag Kp. Kp är endast funktion av T. Behöver Gibbs energi.

    # ln(K_p) = - sum(v_i g_i^o(tilde)) /(tilde(R) T)

    # g = h - Ts

    # h = h_f + h(T)



    return