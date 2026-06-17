from thermo import mixture


def egr_cool(ph_i, th_i, mh, equ_h, pc_i, tc_i, mc, th_o_target):
    """
    Air-to-air heat exchanger using effectiveness-NTU method.
    Solves for effectiveness given a target hot side outlet temperature.

    Args:
        ph_i:         Hot side inlet pressure (Pa)
        th_i:         Hot side inlet temperature (K)
        mh:           Hot side mass flow rate (kg/s)
        pc_i:         Cold side inlet pressure (Pa)
        tc_i:         Cold side inlet temperature (K)
        th_o_target:  Target hot side outlet temperature (K)

    Returns:
        ph_o:          Hot side outlet pressure (Pa)
        th_o:          Hot side outlet temperature (K)
        pc_o:          Cold side outlet pressure (Pa)
        tc_o:          Cold side outlet temperature (K)
        effectiveness: Heat exchanger effectiveness (-)

    Raises:
        ValueError: If th_o_target is outside the physically reachable range.
    """
    if not (tc_i < th_o_target < th_i):
        raise ValueError(
            f"th_o_target={th_o_target:.1f} K must be between "
            f"tc_i={tc_i:.1f} K and th_i={th_i:.1f} K."
        )

    max_iter = 10
    tol = 1e-3

    dp_cold = 0.07
    dp_hot = 0.07

    # th_o is fixed, only tc_o needs an initial guess
    th_o = th_o_target
    tc_o = tc_i + 100

    for _ in range(max_iter):
        tc_o_prev = tc_o

        t_avg_hot  = (th_i + th_o) / 2
        t_avg_cold = (tc_i + tc_o) / 2

        _, _, cp_hot,  *_ = mixture(t_avg_hot,  ph_i, equ_h)
        _, _, cp_cold, *_ = mixture(t_avg_cold, pc_i, 0)

        C_hot    = mh * cp_hot
        C_cold   = mc * cp_cold
        C_min    = min(C_hot, C_cold)

        Q_actual = C_hot * (th_i - th_o)
        tc_o     = tc_i + Q_actual / C_cold

        if abs(tc_o - tc_o_prev) < tol:
            break

    Q_max         = C_min * (th_i - tc_i)
    effectiveness = Q_actual / Q_max

    if not (0 < effectiveness <= 1):
        raise ValueError(
            f"Computed effectiveness={effectiveness:.3f} is outside (0, 1]. "
            "Check input parameters."
        )

    ph_o = ph_i * (1 - dp_hot)
    pc_o = pc_i * (1 - dp_cold)

    return ph_o, pc_o, tc_o, effectiveness, Q_max