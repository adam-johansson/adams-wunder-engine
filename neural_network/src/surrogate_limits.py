from piston_engine.src.misc.seiliger import seiliger
from piston_engine.src.misc.temp_lim import t_in_lim
from thermo import fuel_props
import numpy as np


def input_outside_limits(input_array):
    """
    Simple version that just returns True if any input is outside limits
    """

    input_array = np.atleast_2d(input_array)

    # Extract parameters
    pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far34 = input_array.T

    # Define limits
    p_lim = [1e5, 35e5]
    T_lim = [250, 1000]
    cr_lim = [4, 16]
    d_lim = [0.10, 0.20]
    p_ratio_lim = [0.9, 1.5]
    v_mean_lim = [8, 15]
    fuel_t_lim = [220, 550]

    far_s, _ = fuel_props("jetA")
    far_lim = [far_s / 3.0, far_s / 1.0]

    # Check all limits in one expression
    outside_limits = (np.any((pin < p_lim[0]) | (pin > p_lim[1])) or
                      np.any((Tin < T_lim[0]) | (Tin > T_lim[1])) or
                      np.any((p_ratio < p_ratio_lim[0]) | (p_ratio > p_ratio_lim[1])) or
                      np.any((cr < cr_lim[0]) | (cr > cr_lim[1])) or
                      np.any((bore < d_lim[0]) | (bore > d_lim[1])) or
                      np.any((v_mean < v_mean_lim[0]) | (v_mean > v_mean_lim[1])) or
                      np.any((T_fuel < fuel_t_lim[0]) | (T_fuel > fuel_t_lim[1])) or
                      np.any((far34 < far_lim[0]) | (far34 > far_lim[1])))

    if outside_limits:
        return outside_limits

    # rough estimation of the peak pressure
    pmax_seiliger = seiliger(pin, Tin, cr, far34, bore, "jetA")

    # estimation of the highest possible temperature for given pressure
    t_limit = t_in_lim(pin)

    # t_limit = 1000000
    # if predicted pressure under 400 bar
    if pmax_seiliger > 400 * 1e5 or Tin > t_limit:
        return True

    return outside_limits