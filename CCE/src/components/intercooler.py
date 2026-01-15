from thermo import mixture
import math


def intercool(ph_i, th_i, mh, pc_i, tc_i,  effectiveness, c_ratio):
    """
    Air-to-air heat exchanger using effectiveness-NTU method

    Args:
        ph_i: Hot side inlet pressure (Pa)
        th_i: Hot side inlet temperature (K)
        mh: Hot side mass flow rate (kg/s)
        pc_i: Cold side inlet pressure (Pa)
        tc_i: Cold side inlet temperature (K)

    Returns:
        ph_o: Hot side outlet pressure (Pa)
        th_o: Hot side outlet temperature (K)
        pc_o: Cold side outlet pressure (Pa)
        tc_o: Cold side outlet temperature (K)
        mc: Cold side mass flow rate (kg/s) - calculated for Cr = 1
        effectiveness: Heat exchanger effectiveness
    """

    # Hard coded parameters
    max_iter = 10
    tol = 1e-3

    # pressure losses
    # constant for cold side
    dp_cold = 0.07

    # linearly varying from 0 to 7 percent for hot side

    if c_ratio < 1.0:
        dp_hot = c_ratio * 0.07
    else:
        dp_hot = 0.07


    # Initial guess for outlet temperatures
    th_o = th_i - 30  # Initial guess
    tc_o = tc_i + 30  # Initial guess

    # Iterative solution
    for iteration in range(max_iter):
        # Store previous values for convergence check
        th_o_prev = th_o
        tc_o_prev = tc_o

        # Calculate average temperatures for property evaluation
        t_avg_hot = (th_i + th_o) / 2
        t_avg_cold = (tc_i + tc_o) / 2

        # Get specific heats from mixture function (3rd output)
        _, _, cp_hot, _, _, _, _, _ = mixture(t_avg_hot, ph_i, 0)
        _, _, cp_cold, _, _, _, _, _ = mixture(t_avg_cold, pc_i, 0)

        # Calculate cold mass flow for a given heat capacity ratio
        mc = c_ratio* mh * cp_hot / cp_cold

        # Heat capacity rates
        C_hot = mh * cp_hot  # W/K
        C_cold = mc * cp_cold  # W/K

        C_min = min(C_hot, C_cold)

        # Maximum possible heat transfer
        Q_max = C_min * (th_i - tc_i)

        # Actual heat transfer
        Q_actual = effectiveness * Q_max

        # Calculate new outlet temperatures
        th_o = th_i - Q_actual / C_hot
        tc_o = tc_i + Q_actual / C_cold

        # Check convergence
        temp_change = max(abs(th_o - th_o_prev), abs(tc_o - tc_o_prev))
        if temp_change < tol:
            #print(f"convergence intercooler: {iteration}")
            break

    # Apply pressure drops
    ph_o = ph_i * (1 - dp_hot)
    pc_o = pc_i * (1 - dp_cold)

    return ph_o, th_o, pc_o, tc_o, mc