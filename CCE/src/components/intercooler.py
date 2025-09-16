from thermo import mixture
import math

def intercool(ph_i, th_i, mh, pc_i, tc_i, mc):
    """
    Air-to-air heat exchanger using effectiveness-NTU method

    Args:
        ph_i: Hot side inlet pressure (Pa)
        th_i: Hot side inlet temperature (K)
        mh: Hot side mass flow rate (kg/s)
        pc_i: Cold side inlet pressure (Pa)
        tc_i: Cold side inlet temperature (K)
        mc: Cold side mass flow rate (kg/s)
        UA: Overall heat transfer coefficient × area (W/K)
        config: Flow configuration ('counterflow', 'parallel', 'crossflow')
        dp_hot: Hot side pressure drop fraction (dimensionless)
        dp_cold: Cold side pressure drop fraction (dimensionless)
        max_iter: Maximum iterations for convergence
        tol: Temperature convergence tolerance (K)

    Returns:
        ph_o: Hot side outlet pressure (Pa)
        th_o: Hot side outlet temperature (K)
        pc_o: Cold side outlet pressure (Pa)
        tc_o: Cold side outlet temperature (K)
    """

    # hard coded stuff
    UA = 100
    dp_hot = 0.02
    dp_cold = 0.02
    max_iter = 10
    tol = 1e-3


    # Initial guess for outlet temperatures
    th_o = th_i - 50  # Initial guess
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

        # Heat capacity rates
        C_hot = mh * cp_hot  # W/K
        C_cold = mc * cp_cold  # W/K

        # Minimum and maximum heat capacity rates
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)

        # Heat capacity rate ratio
        Cr = C_min / C_max

        # Number of Transfer Units
        NTU = UA / C_min

        # Calculate effectiveness based (assuming counterflow)
        if abs(Cr - 1.0) < 1e-6:  # Cr = 1
            effectiveness = NTU / (1 + NTU)
        else:
            exp_term = math.exp(-NTU * (1 - Cr))
            effectiveness = (1 - exp_term) / (1 - Cr * exp_term)


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
            print(f"convergence intercooler: {iteration}")
            break

    # Apply pressure drops
    ph_o = ph_i * (1 - dp_hot)
    pc_o = pc_i * (1 - dp_cold)

    #print(f"Effectivness: {effectiveness}")

    return ph_o, th_o, pc_o, tc_o, effectiveness