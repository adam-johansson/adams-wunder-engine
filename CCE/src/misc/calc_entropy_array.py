import numpy as np

from thermo import mixture, fuel_props



def entropy_array(p_arr, T_arr, far_arr, fuel_type):

    far_s, _ = fuel_props(fuel_type)

    equ_arr = far_arr / far_s

    s_arr = np.zeros(len(p_arr))
    for i in range(len(p_arr)):

        p = p_arr[i]
        T = T_arr[i]
        equ = equ_arr[i]
        _, _, _, _, _, _, s, _ = mixture(T, p, equ, fuel_type)
        s_arr[i] = s

    # subtract initial value to see increase in entropy
    s_arr = s_arr - s_arr[1]

    # convert from J/(kg * K) to kJ/(kg K)
    s_arr = s_arr * 1e-3

    return s_arr