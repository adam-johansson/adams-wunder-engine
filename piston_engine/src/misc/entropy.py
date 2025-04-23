import numpy as np

from thermo import mixture


def entropy_calc(t_list, phi_list, fuel_type, p_list):

    entropy_list = []

    for t, phi, p in zip(t_list, phi_list, p_list):

        h, u, cp, cv, R, gamma, entropy, _ = mixture(phi, t, p, fuel_type)
        entropy_list.append(entropy)

    entropy_array = np.array(entropy_list)

    # total entropy (not mass specific)
    # entropy_array = entropy_array * m

    return entropy_array
