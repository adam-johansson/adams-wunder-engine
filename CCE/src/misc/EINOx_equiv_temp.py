from thermo import flame_temp_cea
from scipy.optimize import brentq



def equivalent_temperature(T_start, p_start, equ_start, fuel_type):

    # stoichiometric flame temp for equ_start
    t_flame = flame_temp_cea(T_start, equ_start, fuel_type, p_start, 1.0)
    print(f"t_flame: {t_flame}")
    print(f"Tstart: {T_start}, p_start: {p_start*1e-5}, equ_start: {equ_start}")

    t_flame = flame_temp_cea(300, equ_start, fuel_type, p_start, 1.0)
    print(f"t_flame: {t_flame}")
    print(f"Tstart: {300}, p_start: {p_start*1e-5}, equ_start: {equ_start}")

    t_flame = flame_temp_cea(T_start, 0.0, fuel_type, p_start, 1.0)
    print(f"t_flame: {t_flame}")
    print(f"Tstart: {T_start}, p_start: {p_start*1e-5}, equ_start: {0.0}")

    t_flame = flame_temp_cea(300, 0.0, fuel_type, p_start, 1.0)
    print(f"t_flame: {t_flame}")
    print(f"Tstart: {300}, p_start: {p_start*1e-5}, equ_start: {0.0}")


    def find_equivalent_temp(T):

        # flame temp with pure air before combustion (equ=0)
        t_flame_guess = flame_temp_cea(T, 0.0, fuel_type, p_start, 1.0)
        print(t_flame_guess)

        return t_flame_guess - t_flame
    

    T_equivalent = brentq(find_equivalent_temp, 300, 2000)

    print(f"T_equivalent: {T_equivalent}")
    

    return T_equivalent