import numpy as np
import matplotlib.pyplot as plt


def plot_progress(phi, cycle_phi, P, T, V, V1, m, Q_in):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
    for i in range(it):
        ax1.plot(np.add(phi, i * cycle_phi), P[i] / 1e5)
        ax2.plot(np.add(phi, i * cycle_phi), T[i])
        ax3.plot(np.add(phi, i * cycle_phi), V[i] * 1000)
        ax3.plot(np.add(phi, i * cycle_phi), V1 * 1000)
        ax4.plot(np.add(phi, i * cycle_phi), m[i] / V[i])
        ax5.plot(np.add(phi, i * cycle_phi), Q_in[i])
        ax6.plot(np.add(phi, i * cycle_phi), m[i])

    ax1.set_ylabel(r"p [bar]")
    ax1.set_xticklabels([])
    ax1.grid()

    ax2.set_ylabel(r"T [K]")
    ax2.set_xticklabels([])
    ax2.grid()

    ax3.set_ylabel(r"V [L]")
    ax3.set_xticklabels([])
    ax3.grid()

    ax4.set_ylabel(r"rho [kg/m3]")
    ax4.set_xticklabels([])
    ax4.grid()

    ax5.set_ylabel(r"Qf [J/kg]")
    ax5.set_xticklabels([])
    ax5.grid()

    ax6.set_xlabel("phi [rad]")
    ax6.set_ylabel(r"m [kg]")
    ax6.grid()
    return


def plot_essentials(phi, T, P, m, equ, validation=False):
    if validation:
        # loading validation data
        from piston_engine.src.misc.NASAdata import load_NASA

        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = (
            load_NASA()
        )

    fig, ax3 = plt.subplots()
    ax3.plot(phi * 180 / np.pi, T[-1], label="Our simulation")
    if validation:
        ax3.scatter(ca_NASA, T_NASA, marker="x", label="NASA-validation")
    ax3.set_xlabel("phi [deg]")
    ax3.set_ylabel(r"T [K]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax3.grid()

    fig, ax4 = plt.subplots()
    ax4.plot(phi * 180 / np.pi, P[-1] * 1e-5, label="Our simulation")
    if validation:
        ax4.scatter(ca_NASA, p_NASA, marker="x", label="NASA-validation")
    ax4.set_xlabel("phi [deg]")
    ax4.set_ylabel(r"p [bar]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()
    # ax4.set_xlim(100,250)
    # ax4.set_ylim(5,15)

    # %%
    fig, ax5 = plt.subplots()
    ax5.plot(phi * 180 / np.pi, m[-1], label="Our simulation")
    if validation:
        ax5.scatter(ca_NASA, m_NASA, marker="x", label="NASA-validation")
    ax5.set_xlabel("phi [deg]")
    ax5.set_ylabel(r"m [kg]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax5.grid()

    fig, ax10 = plt.subplots()
    ax10.plot(phi * 180 / np.pi, equ[-1], label="Our simulation")
    if validation:
        ax10.scatter(ca_NASA, phi_NASA, marker="x", label="NASA-validation")
    ax10.set_xlabel("phi [deg]")
    ax10.set_ylabel(r"equivalence ratio [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax10.grid()

    plt.show()
    return


def plot_energy(phi, W, Q, m_in, Q_in, mf):
    fig, ax6 = plt.subplots()
    ax6.plot(phi * 180 / np.pi, W[-1], label="last cycle")
    ax6.set_xlabel("phi [deg]")
    ax6.set_ylabel(r"integral(p*dVdphi) [J]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax6.grid()

    fig, ax7 = plt.subplots()
    ax7.plot(phi * 180 / np.pi, Q[-1], label="last cycle")
    ax7.set_xlabel("phi [deg]")
    ax7.set_ylabel(r"Q [J]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax7.grid()

    fig, ax9 = plt.subplots()
    ax9.plot(phi * 180 / np.pi, Q_in[-1], label="last cycle")
    ax9.set_xlabel("phi [deg]")
    ax9.set_ylabel(r"Q_in [kg]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax9.grid()
    ax9.set_xlim([340, 400])
    plt.show()
    return


def plot_massflows(phi, m_in, mf, mdotin, mdotout, V, validation=False):
    if validation:
        from piston_engine.src.misc.NASAdata import load_NASA

        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = (
            load_NASA()
        )

    fig, ax8 = plt.subplots()
    ax8.plot(phi * 180 / np.pi, m_in[-1], label="last cycle")
    ax8.set_xlabel("phi [deg]")
    ax8.set_ylabel(r"m_in [kg]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax8.grid()

    fig, ax10 = plt.subplots()
    ax10.plot(phi * 180 / np.pi, mf[-1] * 1e3, label="last cycle")
    ax10.set_xlabel("phi [deg]")
    ax10.set_ylabel(r"mf [g]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax10.grid()
    ax10.set_xlim([340, 400])

    fig, ax17 = plt.subplots()
    ax17.plot(phi * 180 / np.pi, mdotin[-1], label="our simulation")
    if validation:
        ax17.scatter(ca_NASA, mdotin_NASA, marker="x", label="NASA-validation")
    ax17.set_xlabel("phi [deg]")
    ax17.set_ylabel(r"mdotin [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax17.grid()

    fig, ax18 = plt.subplots()
    ax18.plot(phi * 180 / np.pi, mdotout[-1], label="our simulation")
    if validation:
        ax18.scatter(ca_NASA, mdotout_NASA, marker="x", label="NASA-validation")
    ax18.set_xlabel("phi [deg]")
    ax18.set_ylabel(r"mdotout [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax18.grid()

    fig, ax19 = plt.subplots()
    ax19.plot(phi * 180 / np.pi, V[-1] * 1000, label="our simulation")
    ax19.set_xlabel("phi [deg]")
    ax19.set_ylabel(r"Vt [l]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax19.grid()

    plt.show()
    return


def plot_manifolds(phi, equ_IP, m_IP, T_IP, equ_EP, m_EP, T_EP):
    fig, ax11 = plt.subplots()
    ax11.plot(phi * 180 / np.pi, equ_IP[-1], label="last cycle")
    ax11.set_xlabel("phi [deg]")
    ax11.set_ylabel(r"equivalence ratio IP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax11.grid()

    fig, ax12 = plt.subplots()
    ax12.plot(phi * 180 / np.pi, m_IP[-1], label="last cycle")
    ax12.set_xlabel("phi [deg]")
    ax12.set_ylabel(r"mass IP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax12.grid()

    fig, ax13 = plt.subplots()
    ax13.plot(phi * 180 / np.pi, T_IP[-1], label="last cycle")
    ax13.set_xlabel("phi [deg]")
    ax13.set_ylabel(r"T IP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax13.grid()

    fig, ax14 = plt.subplots()
    ax14.plot(phi * 180 / np.pi, equ_EP[-1], label="last cycle")
    ax14.set_xlabel("phi [deg]")
    ax14.set_ylabel(r"equivalence ratio EP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax14.grid()

    fig, ax15 = plt.subplots()
    ax15.plot(phi * 180 / np.pi, m_EP[-1], label="last cycle")
    ax15.set_xlabel("phi [deg]")
    ax15.set_ylabel(r"mass EP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax15.grid()

    fig, ax16 = plt.subplots()
    ax16.plot(phi * 180 / np.pi, T_EP[-1], label="last cycle")
    ax16.set_xlabel("phi [deg]")
    ax16.set_ylabel(r"T EP [-]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax16.grid()
    return


def plot_convergence(pdiff, Tdiff, mdiff, equdiff, T_out_diff, m_fuel_diff):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

    ax1.plot(np.abs(pdiff[1:]))
    ax2.plot(np.abs(Tdiff[1:]))
    ax3.plot(np.abs(mdiff[1:]))
    ax4.plot(np.abs(equdiff[1:]))
    ax5.plot(np.abs(T_out_diff[1:]))
    ax6.plot(np.abs(m_fuel_diff[1:]))

    ax1.set_ylabel(r"pdiff [Pa]")
    ax1.set_xticklabels([])
    ax1.grid()
    ax1.set_yscale("log")

    ax2.set_ylabel(r"Tdiff [K]")
    ax2.set_xticklabels([])
    ax2.grid()
    ax2.set_yscale("log")

    ax3.set_ylabel(r"mdiff [mg]")
    ax3.set_xticklabels([])
    ax3.grid()
    ax3.set_yscale("log")

    ax4.set_ylabel(r"equdiff [-]")
    ax4.set_xticklabels([])
    ax4.grid()
    ax4.set_yscale("log")

    ax5.set_ylabel(r"T_out_diff [-]")
    ax5.set_xticklabels([])
    ax5.grid()
    ax5.set_yscale("log")

    ax6.set_ylabel(r"m_fuel_diff [-]")
    ax6.set_xticklabels([])
    ax6.grid()
    ax6.set_yscale("log")

    return


def plot_details(
    phi,
    P,
    T,
    phi_open_in,
    phi_close_in,
    phi_open_out,
    phi_close_out,
    phi_sc,
    validation=False,
):
    p_order = np.roll(P[-1], -np.argwhere(phi * 180 / np.pi > 180)[0][0])
    T_order = np.roll(T[-1], -np.argwhere(phi * 180 / np.pi > 180)[0][0])
    phi_order = phi - phi[0]
    if validation:
        from piston_engine.src.misc.NASAdata import load_NASA

        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = (
            load_NASA()
        )
        p_NASA_order = np.roll(p_NASA, -26)
        T_NASA_order = np.roll(T_NASA, -26)
        ca_NASA_order = ca_NASA - 50

    lw_v = 3
    figsize = (16, 5)

    fig, ax3 = plt.subplots(figsize=figsize)
    ax3.plot(phi_order * 180 / np.pi, T_order, label="Simulation", color="k", lw=2)
    if validation:
        ax3.scatter(
            ca_NASA_order,
            T_NASA_order,
            marker="X",
            label="NASA-validation",
            color="k",
            s=64,
        )
    ax3.axvline(x=phi_open_out * 180 / np.pi + 180, color="b", ls="--", lw=lw_v)
    ax3.axvline(x=phi_open_in * 180 / np.pi + 180, color="r", ls="--", lw=lw_v)
    ax3.axvline(x=phi_close_out * 180 / np.pi - 180, color="g", ls="--", lw=lw_v)
    ax3.axvline(x=phi_close_in * 180 / np.pi - 180, color="k", ls="--", lw=lw_v)
    ax3.axvline(x=phi_sc * 180 / np.pi - 180, color="m", ls="--", lw=lw_v)
    ax3.text(phi_open_out * 180 / np.pi + 180 + 3, 2250, "EVO", fontsize=24)
    ax3.text(phi_open_in * 180 / np.pi + 180 + 3, 2250, "IVO", fontsize=24)
    ax3.text(phi_close_out * 180 / np.pi - 180 + 3, 2250, "EVC", fontsize=24)
    ax3.text(phi_close_in * 180 / np.pi - 180 + 3, 2250, "IVC", fontsize=24)
    ax3.text(phi_sc * 180 / np.pi - 180 + 3, 2250, "SOC", fontsize=24)
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=32)
    ax3.set_ylabel(r"Temperature $T$ [K]", fontsize=32)
    ax3.set_title(r"$T - \theta$ diagram", fontsize=32)
    ax3.set_xlim(45, 405)
    ax3.set_xticks([45, 90, 135, 180, 225, 270, 315, 360, 405])
    ax3.tick_params(labelsize=24)
    plt.legend(loc="best", frameon=True, fontsize=24)
    ax3.grid()

    fig, ax4 = plt.subplots(figsize=figsize)
    ax4.plot(
        phi_order * 180 / np.pi, p_order * 1e-5, label="Simulation", color="k", lw=2
    )
    if validation:
        ax4.scatter(
            ca_NASA_order,
            p_NASA_order,
            marker="X",
            label="NASA-validation",
            color="k",
            s=64,
        )
    ax4.axvline(x=phi_open_out * 180 / np.pi + 180, color="b", ls="--", lw=lw_v)
    ax4.axvline(x=phi_open_in * 180 / np.pi + 180, color="r", ls="--", lw=lw_v)
    ax4.axvline(x=phi_close_out * 180 / np.pi - 180, color="g", ls="--", lw=lw_v)
    ax4.axvline(x=phi_close_in * 180 / np.pi - 180, color="k", ls="--", lw=lw_v)
    ax4.axvline(x=phi_sc * 180 / np.pi - 180, color="m", ls="--", lw=lw_v)
    ax4.text(phi_open_out * 180 / np.pi + 180 + 3, 175, "EVO", fontsize=24)
    ax4.text(phi_open_in * 180 / np.pi + 180 + 3, 175, "IVO", fontsize=24)
    ax4.text(phi_close_out * 180 / np.pi - 180 + 3, 175, "EVC", fontsize=24)
    ax4.text(phi_close_in * 180 / np.pi - 180 + 3, 175, "IVC", fontsize=24)
    ax4.text(phi_sc * 180 / np.pi - 180 + 3, 2250, "SOC", fontsize=24)
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=32)
    ax4.set_ylabel(r"Pressure $p$ [bar]", fontsize=32)
    ax4.set_title(r"$p - \theta$ diagram", fontsize=32)
    ax4.set_xlim(45, 405)
    ax4.set_xticks([45, 90, 135, 180, 225, 270, 315, 360, 405])
    ax4.tick_params(labelsize=24)
    plt.legend(loc="best", frameon=True, fontsize=24)
    ax4.grid()

    plt.show()
    return


def plot_validation_nasa(phi, P, T, m, equ):
    p_order = np.roll(P, np.argwhere((phi - phi[0]) * 180 / np.pi > 100)[0][0])
    T_order = np.roll(T, np.argwhere((phi - phi[0]) * 180 / np.pi > 100)[0][0])
    m_order = np.roll(m, np.argwhere((phi - phi[0]) * 180 / np.pi > 100)[0][0])
    equ_order = np.roll(equ, np.argwhere((phi - phi[0]) * 180 / np.pi > 100)[0][0])
    phi_order = phi - phi[0]

    from piston_engine.src.misc.NASAdata import load_NASA

    ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = load_NASA()
    p_NASA_order = np.roll(p_NASA, 21)
    T_NASA_order = np.roll(T_NASA, 21)
    m_NASA_order = np.roll(m_NASA, 21)
    equ_NASA_order = np.roll(phi_NASA, 21)
    ca_NASA_order = ca_NASA - 105

    fs = 52
    figsize = (20, 16)
    res = 50

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        phi_order * 180 / np.pi, m_order * 1000, label="Piston model", color="k", lw=4
    )
    ax1.scatter(
        ca_NASA_order,
        m_NASA_order * 1000,
        marker="X",
        label="NASA-CR-185155",
        color="r",
        s=512,
    )
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.set_ylabel(r"Mass $m$ [g]", fontsize=fs)
    ax1.set_title(r"$m - \theta$ diagram", fontsize=fs)
    ax1.set_xlim(0, 360)
    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.tick_params(labelsize=fs)
    plt.legend(loc="best", frameon=True, fontsize=fs)
    ax1.grid()
    plt.savefig(
        "piston_engine/simulation_data/figures/m_validation.pdf", dpi=res, bbox_inches="tight"
    )

    fig, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(phi_order * 180 / np.pi, equ_order, label="Piston model", color="k", lw=4)
    ax2.scatter(
        ca_NASA_order,
        equ_NASA_order,
        marker="X",
        label="NASA-CR-185155",
        color="r",
        s=512,
    )
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.set_ylabel(r"Equivalence ratio $\phi$ [-]", fontsize=fs)
    ax2.set_title(r"$\phi - \theta$ diagram", fontsize=fs)
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.tick_params(labelsize=fs)
    plt.legend(loc="best", frameon=True, fontsize=fs)
    ax2.grid()
    plt.savefig(
        "piston_engine/simulation_data/figures/equ_validation.pdf", dpi=res, bbox_inches="tight"
    )

    fig, ax3 = plt.subplots(figsize=figsize)
    ax3.plot(phi_order * 180 / np.pi, T_order, label="Piston model", color="k", lw=4)
    ax3.scatter(
        ca_NASA_order,
        T_NASA_order,
        marker="X",
        label="NASA-CR-185155",
        color="r",
        s=512,
    )
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax3.set_ylabel(r"Temperature $T$ [K]", fontsize=fs)
    ax3.set_title(r"$T - \theta$ diagram", fontsize=fs)
    ax3.set_xlim(0, 360)
    ax3.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax3.tick_params(labelsize=fs)
    plt.legend(loc="best", frameon=True, fontsize=fs)
    ax3.grid()
    plt.savefig(
        "piston_engine/simulation_data/figures/T_validation.pdf", dpi=res, bbox_inches="tight"
    )

    fig, ax4 = plt.subplots(figsize=figsize)
    ax4.plot(
        phi_order * 180 / np.pi, p_order * 1e-5, label="Piston model", color="k", lw=4
    )
    ax4.scatter(
        ca_NASA_order,
        p_NASA_order,
        marker="X",
        label="NASA-CR-185155",
        color="r",
        s=512,
    )
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)
    ax4.set_title(r"$p - \theta$ diagram", fontsize=fs)
    ax4.set_xlim(0, 360)
    ax4.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax4.tick_params(labelsize=fs)
    plt.legend(loc="best", frameon=True, fontsize=fs)
    ax4.grid()
    plt.savefig(
        "piston_engine/simulation_data/figures/p_validation.pdf", dpi=res, bbox_inches="tight"
    )
    plt.show()

    # create output data for tikz
    phi_transpose = np.atleast_2d(phi_order * 180 / np.pi).T
    phi_transpose = phi_transpose[::100]

    m_transpose = np.atleast_2d(m_order * 1000).T
    m_transpose = m_transpose[::100]

    equ_transpose = np.atleast_2d(equ_order).T
    equ_transpose = equ_transpose[::100]

    T_transpose = np.atleast_2d(T_order).T
    T_transpose = T_transpose[::100]

    p_transpose = np.atleast_2d(p_order * 1e-5).T
    p_transpose = p_transpose[::100]

    ca_NASA_transpose = np.atleast_2d(ca_NASA_order).T
    m_NASA_transpose = np.atleast_2d(m_NASA_order * 1000).T
    equ_NASA_transpose = np.atleast_2d(equ_NASA_order).T
    T_NASA_transpose = np.atleast_2d(T_NASA_order).T
    p_NASA_transpose = np.atleast_2d(p_NASA_order).T

    m_sim = np.concatenate((phi_transpose, m_transpose), axis=1)
    m_true = np.concatenate((ca_NASA_transpose, m_NASA_transpose), axis=1)

    equ_sim = np.concatenate((phi_transpose, equ_transpose), axis=1)
    equ_true = np.concatenate((ca_NASA_transpose, equ_NASA_transpose), axis=1)

    T_sim = np.concatenate((phi_transpose, T_transpose), axis=1)
    T_true = np.concatenate((ca_NASA_transpose, T_NASA_transpose), axis=1)

    p_sim = np.concatenate((phi_transpose, p_transpose), axis=1)
    p_true = np.concatenate((ca_NASA_transpose, p_NASA_transpose), axis=1)

    np.savetxt("piston_engine/validation_output_data/NASA/m_validation.dat", m_true, fmt="%.5f")
    np.savetxt("piston_engine/validation_output_data/NASA/m_simulation.dat", m_sim, fmt="%.5f")

    np.savetxt("piston_engine/validation_output_data/NASA/equ_validation.dat", equ_true, fmt="%.5f")
    np.savetxt("piston_engine/validation_output_data/NASA/equ_simulation.dat", equ_sim, fmt="%.5f")

    np.savetxt("piston_engine/validation_output_data/NASA/T_validation.dat", T_true, fmt="%.5f")
    np.savetxt("piston_engine/validation_output_data/NASA/T_simulation.dat", T_sim, fmt="%.5f")

    np.savetxt("piston_engine/validation_output_data/NASA/p_validation.dat", p_true, fmt="%.5f")
    np.savetxt("piston_engine/validation_output_data/NASA/p_simulation.dat", p_sim, fmt="%.5f")

    return


def plot_rohr(phi, Q, Q_in, v, apiston, dtdphi, bore, p, t):
    fs = 24
    fig, ax7 = plt.subplots()
    ax7.plot(phi * 180 / np.pi, Q, label="last cycle")
    ax7.set_xlabel("phi [deg]")
    ax7.set_ylabel(r"Q [J]")
    ax7.set_title(r"heat loss, integrated", fontsize=fs)
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax7.grid()

    fig, ax9 = plt.subplots()
    ax9.plot(phi * 180 / np.pi, Q_in, label="last cycle")
    ax9.set_xlabel("phi [deg]")
    ax9.set_ylabel(r"Q_in [J]")
    ax9.set_title(r"heat addition, integrated", fontsize=fs)
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax9.grid()
    ax9.set_xlim([340, 400])

    phi_deg = phi * 180 / np.pi
    rohr = np.diff(Q_in) / np.diff(phi_deg)
    phi2 = (phi_deg[:-1] + phi_deg[1:]) / 2

    fig, ax10 = plt.subplots()
    ax10.plot(phi2, rohr)
    ax10.set_xlabel("phi [deg]")
    ax10.set_ylabel(r"ROHR [J/CA]")
    ax10.set_title(r"ROHR", fontsize=fs)
    ax10.set_xlim(310, 410)
    ax10.set_ylim(0, 150)
    ax10.set_xticks([310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410])
    ax10.grid()

    area = (v / apiston) * bore * np.pi + apiston * 2  # total cylinder area
    area2 = (area[:-1] + area[1:]) / 2  # avg just to plot

    wall_heatloss_rad = np.diff(Q) / np.diff(phi)  # Joule per radian
    wall_heatloss = wall_heatloss_rad / dtdphi  # Joule per second (Watt)
    wall_heatflux = wall_heatloss / area2  # Watt / m^2
    wall_heatflux = wall_heatflux * 1e-4  # Watt / cm^2

    twall = 500  # wall temperature
    t2 = (t[:-1] + t[1:]) / 2  # avg temperature to fit array length
    alpha = wall_heatflux * 1e4 / (t2 - twall)  # W/(m^2 K) heat transfer coefficient

    phi_deg2 = (phi_deg[:-1] + phi_deg[1:]) / 2

    fig, ax11 = plt.subplots()
    ax11.plot(phi2, wall_heatflux)
    ax11.set_xlabel("phi [deg]")
    ax11.set_ylabel(r"Wall heat flux [W/cm2]")
    ax11.set_title(r"Wall heat flux", fontsize=fs)
    ax11.set_xlim(320, 420)
    ax11.set_xticks([320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420])
    ax11.grid()

    fig, ax12 = plt.subplots()
    ax12.plot(phi_deg, p * 1e-5)
    ax12.set_xlabel("phi [deg]")
    ax12.set_ylabel(r"Cylinder pressure [bar]")
    ax12.set_title(r"Pressure", fontsize=fs)
    ax12.set_xlim(310, 410)
    ax12.set_ylim(0, 60)
    ax12.set_xticks([310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410])
    ax12.grid()

    fig, ax13 = plt.subplots()
    ax13.plot(phi_deg2, alpha)
    ax13.set_xlabel("phi [deg]")
    ax13.set_ylabel(r"Heat transfer coefficient [W/ (m^2 K)]")
    ax13.set_title(r"Heat transfer coefficient", fontsize=fs)
    ax13.set_xlim(320, 420)
    # ax13.set_ylim(0, 60)
    ax13.set_xticks([320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420])
    ax13.grid()
    plt.show()

    # saving heat transfer coefficient to ba able to validate
    # np.savetxt("simulation_data/htc.csv", alpha, delimiter=",")
    # p.savetxt("simulation_data/phi2.csv", phi_deg2, delimiter=",")
    return


def plot_pvts(p, v, t, s, s2):
    # s is calculated from NASA9 polynomials of entropy
    # s2 is calculated with Gibbs equation

    fs = 24
    fig, ax12 = plt.subplots()
    ax12.plot(v * 1e3, p * 1e-5)
    ax12.set_xlabel("Volume [liter]")
    ax12.set_ylabel(r"Cylinder pressure [bar]")
    ax12.set_title(r"pV-diagram", fontsize=fs)
    ax12.grid()

    fig, ax13 = plt.subplots()
    ax13.plot(s, t, label="NASA")
    ax13.set_xlabel("Specific entropy [J/K]")
    ax13.set_ylabel(r"Cylinder temperature [K]")
    ax13.set_title(r"Ts-diagram", fontsize=fs)
    ax13.grid()
    ax13.legend()

    fig, ax14 = plt.subplots()
    ax14.plot(s2, t, label="Gibbs")
    ax14.set_xlabel("Specific entropy [J/K]")
    ax14.set_ylabel(r"Cylinder temperature [K]")
    ax14.set_title(r"Ts-diagram", fontsize=fs)
    ax14.grid()
    ax14.legend()

    plt.show()

    return


def plot_massconservation(m_in, mfuel, mout):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    ax1.plot(m_in)
    ax2.plot(mfuel)
    ax3.plot(mout)
    ax4.plot(np.abs(m_in + mfuel - mout))

    ax1.set_ylabel(r"mass in [kg]")
    ax1.set_xticklabels([])
    ax1.grid()
    # ax1.set_yscale('log')

    ax2.set_ylabel(r"mass fuel in [kg]")
    ax2.set_xticklabels([])
    ax2.grid()
    # ax2.set_yscale('log')

    ax3.set_ylabel(r"mass out [kg]")
    ax3.set_xticklabels([])
    ax3.grid()
    # ax3.set_yscale('log')

    ax4.set_ylabel(r"mass conservation (absolute value) [kg]")
    ax4.set_xticklabels([])
    ax4.grid()
    ax4.set_yscale("log")

    return


def plot_energyconservation(
    enthalpy_in, heat_in, fuel_enthalpy_in, enthalpy_out, heat_out, work_out
):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1)

    energy_in = enthalpy_in + fuel_enthalpy_in
    energy_out = enthalpy_out + work_out + heat_out

    conservation = energy_in - energy_out

    # print(f"Energy conservation: {conservation} [J]")

    ax1.plot(enthalpy_in)
    ax2.plot(heat_in)
    ax3.plot(fuel_enthalpy_in)
    ax4.plot(enthalpy_out)
    ax5.plot(heat_out)
    ax6.plot(work_out)
    ax7.plot(np.abs(conservation))

    ax1.set_ylabel(r"enthalpy in [J]")
    ax1.set_xticklabels([])
    ax1.grid()
    # ax1.set_yscale('log')

    ax2.set_ylabel(r"heat in [J]")
    ax2.set_xticklabels([])
    ax2.grid()
    # ax2.set_yscale('log')

    ax3.set_ylabel(r"fuel enthalpy [J]")
    ax3.set_xticklabels([])
    ax3.grid()
    # ax3.set_yscale('log')

    ax4.set_ylabel(r"enthalpy out [J]")
    ax4.set_xticklabels([])
    ax4.grid()
    # ax4.set_yscale('log')

    ax5.set_ylabel(r"heat out [J]")
    ax5.set_xticklabels([])
    ax5.grid()
    # ax4.set_yscale('log')

    ax6.set_ylabel(r"work out [J]")
    ax6.set_xticklabels([])
    ax6.grid()
    # ax4.set_yscale('log')

    ax7.set_ylabel(r"energy conservation [J]")
    ax7.set_xticklabels([])
    ax7.grid()
    ax7.set_yscale("log")

    plt.show()
    return


def plot_convergence2(mEPdiff, mIPdiff, TEPdiff, TIPdiff):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    ax1.plot(np.abs(mEPdiff[1:]))
    ax2.plot(np.abs(mIPdiff[1:]))
    ax3.plot(np.abs(TEPdiff[1:]))
    ax4.plot(np.abs(TIPdiff[1:]))

    ax1.set_ylabel(r"mEP diff [kg]")
    ax1.set_xticklabels([])
    ax1.grid()
    ax1.set_yscale("log")

    ax2.set_ylabel(r"mIP diff [kg]")
    ax2.set_xticklabels([])
    ax2.grid()
    ax2.set_yscale("log")

    ax3.set_ylabel(r"T EP diff [K]")
    ax3.set_xticklabels([])
    ax3.grid()
    ax3.set_yscale("log")

    ax4.set_ylabel(r"T IP diff [K]")
    ax4.set_xticklabels([])
    ax4.grid()
    ax4.set_yscale("log")

    return


def plot_twozone_full(phi, t1, t2, t, evo, sc):

    # high pressure crank angles
    # phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure pressure curve
    # P_hp = np.array(P[np.argwhere((phi > sc) & (phi < evo))])

    # temperature before sc
    T_before = np.array(t[np.argwhere((phi < sc))])

    # temperature after evo
    T_after = np.array(t[np.argwhere((phi > evo))])

    t1 = np.concatenate((T_before, np.transpose(np.atleast_2d(t1)), T_after), axis=0)

    t2 = np.concatenate((T_before, np.transpose(np.atleast_2d(t2)), T_after), axis=0)

    fig, ax1 = plt.subplots()
    ax1.plot(phi * 180 / np.pi, t, label="Single zone")
    ax1.plot(phi * 180 / np.pi, t1, label="Zone 1")
    ax1.plot(phi * 180 / np.pi, t2, label="Zone 2")

    ax1.set_xlabel("phi [deg]")
    ax1.set_ylabel(r"T [K]")
    plt.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()

    # fig, ax2 = plt.subplots()
    # ax2.plot(phi * 180 / np.pi, dqf, label='Single zone')

    # ax2.set_xlabel('phi [deg]')
    # x2.set_ylabel(r'T [K]')
    # plt.legend(loc='best', fontsize='small', frameon=False)
    # ax2.grid()

    return


def plot_twozone_only(phi_hp, T1, T2, T_hp, m1, m2):

    # plt.plot(phi_hp * 180 / np.pi, qf_hp)
    # plt.title("Heat release rate")

    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    # ax2 = ax1.twinx()
    # ax2.plot(phi_hp * 180 / np.pi, P_hp*1e-5, label="Cylinder pressure")
    ax1.plot(phi_hp * 180 / np.pi, T1, label="Zone 1")
    ax1.plot(phi_hp * 180 / np.pi, T2, label="Zone 2")
    ax1.plot(phi_hp * 180 / np.pi, T_hp, label="Single zone")

    ax1.set_xlabel("Crank angle [deg]")
    # ax1.set_ylabel('Cylinder pressure [bar]', color='g')
    ax1.set_ylabel("Temperature [K]", color="b")

    # ax2.legend()
    ax1.legend()

    # m_total is just the sum of zone 1 and zone 2
    m_hp = m1 + m2

    # plot masses
    fig, ax2 = plt.subplots()

    ax2.plot(phi_hp * 180 / np.pi, m1, label="Zone 1")
    ax2.plot(phi_hp * 180 / np.pi, m2, label="Zone 2")
    ax2.plot(phi_hp * 180 / np.pi, m_hp, label="Single zone")

    ax2.set_xlabel("Crank angle [deg]")
    ax2.set_ylabel("Mass [kg]", color="b")

    ax2.legend()

    plt.show()
    return


def plot_twozone_validation(phi, t1, t2, t, p, evo, sc):

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # load data from Heider
    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(dirname, "../../validation_output_data/Heider/p.txt")
    filename_T0 = os.path.join(dirname, "../../validation_output_data/Heider/T_0.txt")
    filename_T1 = os.path.join(dirname, "../../validation_output_data/Heider/T_1.txt")
    filename_T2 = os.path.join(dirname, "../../validation_output_data/Heider/T_2.txt")
    p_heider = np.loadtxt(filename_p, delimiter=",")
    T0_heider = np.loadtxt(filename_T0, delimiter=",")
    T1_heider = np.loadtxt(filename_T1, delimiter=",")
    T2_heider = np.loadtxt(filename_T2, delimiter=",")

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()

    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_heider[:, 0],
        p_heider[:, 1],
        label="p validation",
        color="k",
        lw=1,
        marker="o",
    )

    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(300, evo * 180 / np.pi)
    ax1.set_ylim(-50, 100)
    ax1.set_xticks([300, 360, 420, 480])
    ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)

    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(phi * 180 / np.pi, t, label="Single zone", color="k", lw=1)
    ax2.plot(phi_hp * 180 / np.pi, t1, label="T zone 2", color="g")
    ax2.plot(phi_hp * 180 / np.pi, t2, label="T zone 1", color="b")


    ax2.plot(
        T0_heider[:, 0],
        T0_heider[:, 1],
        label="0 dim validation",
        color="r",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T1_heider[:, 0],
        T1_heider[:, 1],
        label="Zone 1 validation",
        color="g",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T2_heider[:, 0],
        T2_heider[:, 1],
        label="Zone 2 validation",
        color="b",
        lw=1,
        marker="o",
    )

    ax2.set_ylabel(r"Temperature $T$ [K]]")
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]")
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(300, evo * 180 / np.pi)
    ax2.set_ylim(500, 3500)
    ax2.set_xticks([300, 360, 420, 480])
    ax2.set_yticks([1000, 2000, 3000])
    #ax2.tick_params(labelsize=fs)

    # ax3.scatter(ca_NASA_order, T_NASA_order, marker="X", label='NASA-CR-185155', color="r", s=512)

    # plt.legend(loc='best', frameon=True, fontsize=fs)
    # plt.savefig('simulation_data/two_zone_validation/T_p_validation.pdf', dpi=res, bbox_inches='tight')

    # fig, ax2 = plt.subplots()
    # ax2.plot(phi * 180 / np.pi, dqf, label='Single zone')

    # ax2.set_xlabel('phi [deg]')
    # x2.set_ylabel(r'T [K]')
    # plt.legend(loc='best', fontsize='small', frameon=False)
    # ax2.grid()

    return


def plot_no_validation(no, phi):
    """

    :param no: concentration of NO in ppm
    :param phi: crank angle in radians
    :return:
    """

    # phi = phi[1:]

    ca = np.ndarray.flatten(phi * 180 / np.pi)
    # d ppm /dca
    dnodca = np.gradient(no, ca)

    # dnomoldca = np.gradient(no_mol, ca)

    # load data from Heider
    import os

    dirname = os.path.dirname(__file__)
    filename_no = os.path.join(dirname, "../../validation_output_data/Heider/no.txt")
    filename_dnodca = os.path.join(
        dirname, "../../validation_output_data/Heider/dnodca.txt"
    )

    no_heider = np.loadtxt(filename_no, delimiter=",")
    dnodca_heider = np.loadtxt(filename_dnodca, delimiter=",")

    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    lns1 = ax1.plot(phi * 180 / np.pi, no, color="red", label="NO concentration")
    lns2 = ax2.plot(phi * 180 / np.pi, dnodca, color="blue", label="dNOdphi")


    lns3 = ax1.plot(
        no_heider[:, 0], no_heider[:, 1], color="red", label="NO validation", marker="x"
    )
    lns4 = ax2.plot(
        dnodca_heider[:, 0],
        dnodca_heider[:, 1],
        color="blue",
        label="dNOdt validation",
        marker="x",
    )


    ax1.set_xlim(355, 400)

    # set which axis to which side
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")
    #ax1.set_title("NO production")
    ax1.set_ylabel(" NO concentration [ppm]") #mass based
    ax2.set_ylabel("NO production [ppm/ deg]")
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]")
    # ax1.legend(loc='upper right', fontsize='small', frameon=False)

    plt.show()

    return


def plot_nox_diesel_validation(phi, t1, t2, t, p, evo, sc, mf, no):

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # load data from Rakopoulos
    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(dirname, "../../validation_output_data/NO_diesel_val_data/p.txt")
    filename_T0 = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/T_0.txt"
    )
    filename_T1 = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/T_1.txt"
    )
    filename_fuel = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/fuel.txt"
    )
    filename_NO = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/NO_80.txt"
    )
    p_val = np.loadtxt(filename_p, delimiter=",")
    T0_val = np.loadtxt(filename_T0, delimiter=",")
    T1_val = np.loadtxt(filename_T1, delimiter=",")
    fuel_val = np.loadtxt(filename_fuel, delimiter=",")
    no_val = np.loadtxt(filename_NO, delimiter=",")

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()

    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_val[:, 0], p_val[:, 1], label="p validation", color="k", lw=1, marker="o"
    )

    ax1.set_title("-20 deg injection timing")
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(300, evo * 180 / np.pi)
    ax1.set_ylim(-50, 100)
    ax1.set_xticks([300, 360, 420, 480])
    ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)

    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(phi * 180 / np.pi, t, label="Single zone", color="k", lw=1)
    ax2.plot(phi_hp * 180 / np.pi, t1, label="T zone 1", color="g")

    ax2.plot(
        T0_val[:, 0],
        T0_val[:, 1],
        label="0 dim validation",
        color="r",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T1_val[:, 0],
        T1_val[:, 1],
        label="Zone 1 validation",
        color="g",
        lw=1,
        marker="o",
    )

    ax2.set_title("-20 deg injection timing")
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.set_ylabel(r"Temperature $T$ [K]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(300, evo * 180 / np.pi)
    ax2.set_ylim(500, 3500)
    ax2.set_xticks([300, 360, 420, 480])
    ax2.set_yticks([1000, 2000, 3000])
    ax2.tick_params(labelsize=fs)

    fig, ax3 = plt.subplots()
    ax3.plot(phi * 180 / np.pi, mf * 1e6, label="Burned fuel", color="k", lw=2)
    ax3.plot(
        fuel_val[:, 0], fuel_val[:, 1], label="Validation", color="b", lw=2, marker="o"
    )

    ax3.set_xlim(340, 400)
    ax3.set_title("-20 deg injection timing")
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax3.set_ylabel(r"Burned fuel (mg)", fontsize=fs)
    ax3.legend(loc="best", fontsize="small", frameon=False)
    ax3.grid()

    # ax2.plot(phi_hp * 180 / np.pi, t1, label='T zone 1', color="g")

    fig, ax4 = plt.subplots()
    ax4.plot(phi_hp * 180 / np.pi, no, label="NO concentration", color="k", lw=2)
    ax4.plot(
        no_val[:, 0], no_val[:, 1], label="Validation", color="b", lw=2, marker="o"
    )

    ax4.set_xlim(260, 460)
    ax4.set_title("-20 deg injection timing")
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax4.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()

    # convert to deg
    phi_hp = phi_hp * 180 / np.pi

    no_sim = np.hstack((phi_hp, np.transpose(np.atleast_2d(no))))

    phi_min = no_val[0, 0]
    phi_max = no_val[-1, 0]

    no_sim = no_sim[(no_sim[:, 0] > phi_min) & (no_sim[:, 0] < phi_max)]

    # take only every tenth data point
    no_sim = no_sim[::10]

    nos_sim_interpolated = np.interp(no_val[:,0], no_sim[:,0], no_sim[:,1])

    nox_early = np.hstack((np.transpose(np.atleast_2d(nos_sim_interpolated)), no_val))

    np.savetxt("./piston_engine/validation_output_data/NO_diesel_val_output/nox_early.dat", nox_early, fmt="%.5f")

    #plt.show()
    return


def plot_nox_diesel_validation_late(phi, t1, t2, t, p, evo, sc, mf, no):

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # load data from Rakopoulos
    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/p_late.txt"
    )
    filename_T0 = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/T_0_late.txt"
    )
    filename_T1 = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/T_1_late.txt"
    )
    filename_fuel = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/fuel_late.txt"
    )
    filename_NO = os.path.join(
        dirname, "../../validation_output_data/NO_diesel_val_data/NO_80_late.txt"
    )
    p_val = np.loadtxt(filename_p, delimiter=",")
    T0_val = np.loadtxt(filename_T0, delimiter=",")
    T1_val = np.loadtxt(filename_T1, delimiter=",")
    fuel_val = np.loadtxt(filename_fuel, delimiter=",")
    no_val = np.loadtxt(filename_NO, delimiter=",")

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()

    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_val[:, 0], p_val[:, 1], label="p validation", color="k", lw=1, marker="o"
    )
    ax1.set_title("-15 deg injection timing")

    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(300, evo * 180 / np.pi)
    ax1.set_ylim(-50, 100)
    ax1.set_xticks([300, 360, 420, 480])
    ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)

    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(phi * 180 / np.pi, t, label="Single zone", color="k", lw=1)
    ax2.plot(phi_hp * 180 / np.pi, t1, label="T zone 1", color="g")

    ax2.plot(
        T0_val[:, 0],
        T0_val[:, 1],
        label="0 dim validation",
        color="r",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T1_val[:, 0],
        T1_val[:, 1],
        label="Zone 1 validation",
        color="g",
        lw=1,
        marker="o",
    )

    ax2.set_title("-15 deg injection timing")
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.set_ylabel(r"Temperature $T$ [K]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(300, evo * 180 / np.pi)
    ax2.set_ylim(500, 3500)
    ax2.set_xticks([300, 360, 420, 480])
    ax2.set_yticks([1000, 2000, 3000])
    ax2.tick_params(labelsize=fs)

    fig, ax3 = plt.subplots()
    ax3.plot(phi * 180 / np.pi, mf * 1e6, label="Burned fuel", color="k", lw=2)
    ax3.plot(
        fuel_val[:, 0], fuel_val[:, 1], label="Validation", color="b", lw=2, marker="o"
    )

    ax3.set_xlim(340, 400)
    ax3.set_title("-15 deg injection timing")
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax3.set_ylabel(r"Burned fuel (mg)", fontsize=fs)
    ax3.legend(loc="best", fontsize="small", frameon=False)
    ax3.grid()

    # ax2.plot(phi_hp * 180 / np.pi, t1, label='T zone 1', color="g")

    fig, ax4 = plt.subplots()
    ax4.plot(phi_hp * 180 / np.pi, no, label="NO concentration", color="k", lw=2)
    ax4.plot(
        no_val[:, 0], no_val[:, 1], label="Validation", color="b", lw=2, marker="o"
    )

    ax4.set_xlim(260, 460)
    ax4.set_title("-15 deg injection timing")
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax4.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()

    # convert to deg
    phi_hp = phi_hp * 180 / np.pi

    no_sim = np.hstack((phi_hp, np.transpose(np.atleast_2d(no))))

    phi_min = no_val[0, 0]
    phi_max = no_val[-1, 0]

    no_sim = no_sim[(no_sim[:, 0] > phi_min) & (no_sim[:, 0] < phi_max)]

    # take only every tenth data point
    no_sim = no_sim[::10]

    nos_sim_interpolated = np.interp(no_val[:,0], no_sim[:,0], no_sim[:,1])

    nox_late = np.hstack((np.transpose(np.atleast_2d(nos_sim_interpolated)), no_val))

    np.savetxt("./piston_engine/validation_output_data/NO_diesel_val_output/nox_late.dat", nox_late, fmt="%.5f")

    #plt.show()

    return


def plot_scania_highload(phi, p, mf, LHV, Q_apparent):

    # load data from Diotavelli
    import os

    dirname = os.path.dirname(__file__)
    filename_p_low = os.path.join(
        dirname, "../../validation_output_data/Scania/low_load.txt"
    )
    filename_p_high = os.path.join(
        dirname, "../../validation_output_data/Scania/high_load.txt"
    )
    filename_heat_high = os.path.join(
        dirname, "../../validation_output_data/Scania/heat_high.txt"
    )
    p_low_val = np.loadtxt(filename_p_low, delimiter=",")
    p_high_val = np.loadtxt(filename_p_high, delimiter=",")
    heat_high_val = np.loadtxt(filename_heat_high, delimiter=",")

    # get apparent rate of heat relase
    Q_apparent = np.gradient(Q_apparent, phi)

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_high_val[:, 0],
        p_high_val[:, 1],
        label="high load validation",
        color="b",
        lw=1,
        marker="o",
    )
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(
        phi * 180 / np.pi, mf * LHV * np.pi / 180, label="simulation", color="r", lw=1
    )
    ax2.plot(
        heat_high_val[:, 0],
        heat_high_val[:, 1],
        label="validation",
        color="k",
        lw=1,
        marker="o",
    )
    # ax2.plot(phi * 180 / np.pi, Q_apparent * np.pi / 180, label='Apparent', color="b", lw=1)
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(350, 420)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax2.tick_params(labelsize=fs)
    ax2.set_ylabel(r"Heat release [J/deg]", fontsize=fs)

    return


def plot_scania_lowload(phi, p, mf, LHV, Q_apparent):

    # load data from Diotavelli
    import os

    dirname = os.path.dirname(__file__)
    filename_p_low = os.path.join(
        dirname, "../../validation_output_data/Scania/low_load.txt"
    )
    filename_heat_low = os.path.join(
        dirname, "../../validation_output_data/Scania/heat_low.txt"
    )
    p_low_val = np.loadtxt(filename_p_low, delimiter=",")
    heat_low_val = np.loadtxt(filename_heat_low, delimiter=",")

    # get apparent rate of heat relase
    Q_apparent = np.gradient(Q_apparent, phi)

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_low_val[:, 0],
        p_low_val[:, 1],
        label="low load validation",
        color="k",
        lw=1,
        marker="o",
    )
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(
        phi * 180 / np.pi, mf * LHV * np.pi / 180, label="simulation", color="r", lw=1
    )
    ax2.plot(
        heat_low_val[:, 0],
        heat_low_val[:, 1],
        label="validation",
        color="k",
        lw=1,
        marker="o",
    )
    # ax2.plot(phi * 180 / np.pi, Q_apparent * np.pi / 180, label='Apparent', color="b", lw=1)
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(350, 420)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax2.tick_params(labelsize=fs)
    ax2.set_ylabel(r"Heat realease [J/deg]", fontsize=fs)

    return


def val_water_paper_h2(phi, p, mf, LHV, Q_apparent):

    # load data from the H2 water paper
    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(
        dirname, "../../validation_output_data/H2_water/pressure.txt"
    )
    filename_heat = os.path.join(
        dirname, "../../validation_output_data/H2_water/heat.txt"
    )
    p_val = np.loadtxt(filename_p, delimiter=",")
    heat_val = np.loadtxt(filename_heat, delimiter=",")

    # get apparent rate of heat relase
    Q_apparent = np.gradient(Q_apparent, phi)

    pressure = np.array([phi * 180 / np.pi, p])

    # Find the index where time is greater than or equal to the threshold
    index_threshold = np.where(pressure[0] >= 720)[0][0]

    pressure[0, index_threshold:] = pressure[0, index_threshold:] - 720

    # Rearrange the data so all values for time after the threshold are placed first
    pressure = np.concatenate(
        (pressure[:, index_threshold:], pressure[:, :index_threshold]), axis=1
    )

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(pressure[0, :], pressure[1, :] * 1e-5, label="p", color="r", lw=1)
    ax1.plot(p_val[:, 0], p_val[:, 1], label="validation", color="b", lw=1, marker="o")
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    # ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    # ax2.plot(phi * 180 / np.pi, mf * LHV * np.pi / 180, label='simulation', color="r", lw=1)
    ax2.plot(
        heat_val[:, 0], heat_val[:, 1], label="validation", color="k", lw=1, marker="o"
    )
    ax2.plot(
        phi * 180 / np.pi, Q_apparent * np.pi / 180, label="Apparent", color="b", lw=1
    )

    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(355, 380)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax2.tick_params(labelsize=fs)
    ax2.set_ylabel(r"Heat release [J/deg]", fontsize=fs)

    plt.show()

    return


def plot_no(phi, evo, sc, no):

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    fs = 18

    fig, ax4 = plt.subplots()
    ax4.plot(phi_hp * 180 / np.pi, no, label="NO concentration", color="k", lw=2)

    # ax4.set_xlim(260, 460)
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax4.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()

    # plt.show()

    return

def plot_no_with_equi(phi, evo, sc, no, no_equ):

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    fs = 18

    fig, ax4 = plt.subplots()
    ax4.plot(phi_hp * 180 / np.pi, no, label="NO concentration", color="k", lw=2)
    ax4.plot(phi_hp * 180 / np.pi, no_equ, label="NO equilibrium", color="r", lw=2)

    # ax4.set_xlim(260, 460)
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax4.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()

    # plt.show()

    return


def plot_addedfuel(phi, dmfdphi):
    from scipy.integrate import cumulative_trapezoid

    # high pressure crank angles
    # phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    fs = 18

    fuel = cumulative_trapezoid(dmfdphi, phi, initial=0) * 1e6

    # we want phi where 50% of fuel is added. for chalmers h2 engine it is supposed to be at 368 deg

    fig, ax4 = plt.subplots()
    ax4.plot(phi * 180 / np.pi, fuel, color="k", lw=2)

    ax4.set_xlim(300, 425)
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.set_ylabel(r"Added fuel (mg)", fontsize=fs)
    ax4.grid()

    # plt.show()

    return


def val_newcastle(phi, p):

    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(
        dirname, "../../validation_output_data/newcastle/pressure.txt"
    )
    p_val = np.loadtxt(filename_p, delimiter=",")

    pressure = np.array([phi * 180 / np.pi, p])

    # Find the index where time is greater than or equal to the threshold
    index_threshold = np.where(pressure[0] >= 720)[0][0]

    pressure[0, index_threshold:] = pressure[0, index_threshold:] - 720

    # Rearrange the data so all values for time after the threshold are placed first
    pressure = np.concatenate(
        (pressure[:, index_threshold:], pressure[:, :index_threshold]), axis=1
    )

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(pressure[0, :], pressure[1, :] * 1e-5, label="p", color="r", lw=1)
    ax1.plot(p_val[:, 0], p_val[:, 1], label="validation", color="b", lw=1, marker="o")
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    # ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    plt.show()

    return


def val_hcci(phi, p):

    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(
        dirname, "../../validation_output_data/newcastle/pressure.txt"
    )
    p_val = np.loadtxt(filename_p, delimiter=",")

    pressure = np.array([phi * 180 / np.pi, p])

    # Find the index where time is greater than or equal to the threshold
    index_threshold = np.where(pressure[0] >= 720)[0][0]

    pressure[0, index_threshold:] = pressure[0, index_threshold:] - 720

    # Rearrange the data so all values for time after the threshold are placed first
    pressure = np.concatenate(
        (pressure[:, index_threshold:], pressure[:, :index_threshold]), axis=1
    )

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(pressure[0, :], pressure[1, :] * 1e-5, label="p", color="r", lw=1)
    ax1.plot(p_val[:, 0], p_val[:, 1], label="validation", color="b", lw=1, marker="o")
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    # ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    plt.show()

    return


def val_chalmers(phi, p, T, Q, V, equ, premixed):

    import os

    from thermo import mixture

    dirname = os.path.dirname(__file__)

    filename_p = os.path.join(
        dirname, "../../validation/chalmers_h2/case3/p_val.txt"
    )

    filename_T = os.path.join(
        dirname, "../../validation/chalmers_h2/case3/T_val.txt"
    )

    filename_hrr = os.path.join(
        dirname, "../../validation/chalmers_h2/case3/hrr_val.txt"
    )

    p_exp = np.loadtxt(filename_p, delimiter=",")
    T_exp = np.loadtxt(filename_T, delimiter=",")
    hrr_exp = np.loadtxt(filename_hrr, delimiter=",")

    # convert to degrees from radian
    p_sim = np.array([phi * 180 / np.pi, p])
    T_sim = np.array([phi * 180 / np.pi, T])
    Q_sim = np.array([phi * 180 / np.pi, Q])
    V_sim = np.array([phi * 180 / np.pi, V])
    equ_sim = np.array([phi * 180 / np.pi, equ])


    # apparent rate of heat release calculations from here
    # change in pressure
    dp = np.zeros(np.shape(p_sim))
    dp[0, :] = p_sim[0, :]
    dp[1, :] = np.gradient(p_sim[1, :], p_sim[0, :])

    # change in volume
    dV = np.zeros(np.shape(V_sim))
    dV[0, :] = V_sim[0, :]
    dV[1, :] = np.gradient(V_sim[1, :], V_sim[0, :])

    # heat capacity ratio
    # copy just to get the correct size of array
    gamma = np.zeros(np.shape(V_sim))
    gamma[0,:] = V_sim[0,:]

    for i in range(np.shape(gamma)[1]):
        t = T_sim[1, i]
        p_temp = p_sim[1, i]
        equ_temp = equ_sim[1, i]
        _, _, _, _, _, gamma_temp, _, _ = mixture(t, p_temp, equivalence_ratio=0, fuel_type="H2",
                                                  include_fuel_in_reactants=premixed, fuel_air_equ_ratio=equ_temp)
        gamma[1, i] = gamma_temp


    # change in gamma
    dgamma = np.zeros(np.shape(gamma))
    dgamma[0, :] = gamma[0, :]
    dgamma[1, :] = np.gradient(gamma[1, :], gamma[0, :])

    # apparent rate of heat release (AHRR)
    ahrr = np.zeros(np.shape(gamma))
    ahrr[0, :] = gamma[0, :]
    ahrr[1, :] = ((V_sim[1,:] * dp[1,:] + gamma[1,:] * p_sim[1,:] * dV[1,:]) / (gamma[1,:] - 1) -
            p_sim[1,:] * V_sim[1,:] / (gamma[1,:] - 1)**2 * dgamma[1,:])


    # Find the index where time is greater than or equal to the threshold
    index_threshold = np.where(p_sim[0] >= 720)[0][0]

    # adjust angles to go from 0 to 720
    p_sim[0, index_threshold:] = p_sim[0, index_threshold:] - 720
    T_sim[0, index_threshold:] = T_sim[0, index_threshold:] - 720
    Q_sim[0, index_threshold:] = Q_sim[0, index_threshold:] - 720
    V_sim[0, index_threshold:] = V_sim[0, index_threshold:] - 720
    equ_sim[0, index_threshold:] = equ_sim[0, index_threshold:] - 720
    ahrr[0, index_threshold:] = ahrr[0, index_threshold:] - 720

    # make the added heat 0 before heat is added (data points that were in the end before rearanging)
    Q_sim[1, index_threshold:] = 0.0

    # Rearrange the data so all values for time after the threshold are placed first
    p_sim = np.concatenate(
        (p_sim[:, index_threshold:], p_sim[:, :index_threshold]), axis=1
    )

    T_sim = np.concatenate(
        (T_sim[:, index_threshold:], T_sim[:, :index_threshold]), axis=1
    )

    Q_sim = np.concatenate(
        (Q_sim[:, index_threshold:], Q_sim[:, :index_threshold]), axis=1
    )

    V_sim = np.concatenate(
        (V_sim[:, index_threshold:], V_sim[:, :index_threshold]), axis=1
    )

    equ_sim = np.concatenate(
        (equ_sim[:, index_threshold:], equ_sim[:, :index_threshold]), axis=1
    )

    ahrr = np.concatenate(
        (ahrr[:, index_threshold:], ahrr[:, :index_threshold]), axis=1
    )

    # adjust angles to go from -360 to 360
    p_sim[0, :] = p_sim[0, :] - 360
    T_sim[0, :] = T_sim[0, :] - 360
    Q_sim[0, :] = Q_sim[0, :] - 360
    V_sim[0, :] = V_sim[0, :] - 360
    equ_sim[0, :] = equ_sim[0, :] - 360
    ahrr[0, :] = ahrr[0, :] - 360


    # real rate of heat release
    hrr_sim = np.zeros(np.shape(Q_sim))
    hrr_sim[0, :] = Q_sim[0, :]
    hrr_sim[1, :] = np.gradient(Q_sim[1, :], Q_sim[0, :])




    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()
    ax1.plot(p_sim[0, :], p_sim[1, :] * 1e-5, label="simulation", color="r", lw=1)
    ax1.plot(p_exp[:, 1], p_exp[:, 0], label="experiment", color="b", lw=1)
    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    # ax1.set_xlim(260, 510)
    # ax1.set_ylim(-50, 100)
    # ax1.set_xticks([300, 360, 420, 480])
    # ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)
    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(T_sim[0, :], T_sim[1, :], label="simulation", color="r", lw=1)
    ax2.plot(T_exp[:, 1], T_exp[:, 0], label="experiment", color="b", lw=1)
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.tick_params(labelsize=fs)
    ax2.set_ylabel(r"Temperature $T$ [K]", fontsize=fs)

    fig, ax3 = plt.subplots()
    ax3.plot(hrr_sim[0, :], hrr_sim[1, :], label="simulation", color="r", lw=1)
    ax3.plot(hrr_exp[:, 1], hrr_exp[:, 0], label="experiment", color="b", lw=1)
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax3.legend(loc="best", fontsize="small", frameon=False)
    ax3.grid()
    ax3.tick_params(labelsize=fs)
    ax3.set_ylabel(r"HRR", fontsize=fs)
    ax3.set_xlim(-142, 145)

    fig, ax4 = plt.subplots()
    ax4.plot(ahrr[0, :], ahrr[1, :], label="simulation", color="r", lw=1)
    ax4.plot(hrr_exp[:, 1], hrr_exp[:, 0], label="experiment", color="b", lw=1)
    ax4.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax4.legend(loc="best", fontsize="small", frameon=False)
    ax4.grid()
    ax4.tick_params(labelsize=fs)
    ax4.set_ylabel(r"AHRR", fontsize=fs)
    #ax4.set_xlim(-142, 145)

    plt.show()

    # case 2
    #print(f"Validation: eta: 42.5%, power: 13.7kW, IMEP: 11.5 bar, NOX: {261} ppm, T_out: 799 K ")
    #print(f"Validation fuel flow: {0.27} g/s")

    #case 3
    print(f"Validation: eta: 40.1%, power: 6.7kW, IMEP: 8.5 bar, NOX: {3617} ppm, T_out: 798 K ")
    print(f"Validation fuel flow: {0.14} g/s")

    return

def plot_twozone_validation_final(phi, t1, t2, t, p, evo, sc, no, phi_z1):


    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # load data from Heider
    import os

    dirname = os.path.dirname(__file__)
    filename_p = os.path.join(dirname, "../../validation_output_data/Heider_new/p.txt")
    filename_T0 = os.path.join(dirname, "../../validation_output_data/Heider_new/T_0.txt")
    filename_T1 = os.path.join(dirname, "../../validation_output_data/Heider_new/T_1.txt")
    filename_T2 = os.path.join(dirname, "../../validation_output_data/Heider_new/T_2.txt")
    p_heider = np.loadtxt(filename_p, delimiter=",")
    T0_heider = np.loadtxt(filename_T0, delimiter=",")
    T1_heider = np.loadtxt(filename_T1, delimiter=",")
    T2_heider = np.loadtxt(filename_T2, delimiter=",")

    # fs = 52
    fs = 18
    figsize = (20, 16)
    res = 50

    # fig, ax1 = plt.subplots(figsize=figsize)
    fig, ax1 = plt.subplots()

    ax1.plot(phi * 180 / np.pi, p * 1e-5, label="p", color="r", lw=1)
    ax1.plot(
        p_heider[:, 0],
        p_heider[:, 1],
        label="p validation",
        color="k",
        lw=1,
        marker="o",
    )

    ax1.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]", fontsize=fs)
    ax1.legend(loc="best", fontsize="small", frameon=False)
    ax1.grid()
    ax1.set_xlim(300, evo * 180 / np.pi)
    ax1.set_ylim(-50, 100)
    ax1.set_xticks([300, 360, 420, 480])
    ax1.set_yticks([0, 50, 100])
    ax1.tick_params(labelsize=fs)

    ax1.set_ylabel(r"Pressure $p$ [bar]", fontsize=fs)

    fig, ax2 = plt.subplots()
    ax2.plot(phi * 180 / np.pi, t, label="Single zone", color="k", lw=1)
    ax2.plot(phi_hp * 180 / np.pi, t1, label="T zone 2", color="g")
    ax2.plot(phi_hp * 180 / np.pi, t2, label="T zone 1", color="b")


    ax2.plot(
        T0_heider[:, 0],
        T0_heider[:, 1],
        label="0 dim validation",
        color="r",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T1_heider[:, 0],
        T1_heider[:, 1],
        label="Zone 1 validation",
        color="g",
        lw=1,
        marker="o",
    )
    ax2.plot(
        T2_heider[:, 0],
        T2_heider[:, 1],
        label="Zone 2 validation",
        color="b",
        lw=1,
        marker="o",
    )

    ax2.set_ylabel(r"Temperature $T$ [K]]")
    ax2.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]")
    ax2.legend(loc="best", fontsize="small", frameon=False)
    ax2.grid()
    ax2.set_xlim(300, evo * 180 / np.pi)
    ax2.set_ylim(500, 3500)
    ax2.set_xticks([300, 360, 420, 480])
    ax2.set_yticks([1000, 2000, 3000])

    # phi = phi[1:]

    ca = np.ndarray.flatten(phi_z1 * 180 / np.pi)
    # d ppm /dca
    dnodca = np.gradient(no, ca)

    # dnomoldca = np.gradient(no_mol, ca)

    # load data from Heider
    import os

    dirname = os.path.dirname(__file__)
    filename_no = os.path.join(dirname, "../../validation_output_data/Heider_new/no.txt")
    filename_dnodca = os.path.join(
        dirname, "../../validation_output_data/Heider_new/dnodca.txt"
    )

    no_heider = np.loadtxt(filename_no, delimiter=",")
    dnodca_heider = np.loadtxt(filename_dnodca, delimiter=",")

    # plot temperatures and pressure
    fig, ax3 = plt.subplots()

    ax4 = ax3.twinx()

    lns1 = ax3.plot(phi_z1 * 180 / np.pi, no, color="red", label="NO concentration")
    lns2 = ax4.plot(phi_z1 * 180 / np.pi, dnodca, color="blue", label="dNOdphi")


    lns3 = ax3.plot(
        no_heider[:, 0], no_heider[:, 1], color="red", label="NO validation", marker="x"
    )
    lns4 = ax4.plot(
        dnodca_heider[:, 0],
        dnodca_heider[:, 1],
        color="blue",
        label="dNOdt validation",
        marker="x",
    )


    ax3.set_xlim(355, 400)

    # set which axis to which side
    ax3.yaxis.tick_left()
    ax4.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc="upper right")
    #ax1.set_title("NO production")
    ax3.set_ylabel(" NO concentration [ppm]") #mass based
    ax4.set_ylabel("NO production [ppm / deg]")
    ax3.set_xlabel(r"Crank angle $\theta$ [$^{\circ}$]")
    # ax1.legend(loc='upper right', fontsize='small', frameon=False)



    filename_T1_detailed = os.path.join(dirname, "../../validation_output_data/Heider_new/T_1_detailed.txt")
    T1_heider_detailed = np.loadtxt(filename_T1_detailed, delimiter=",")

    fig, ax5 = plt.subplots()

    ax5.plot(phi_hp * 180 / np.pi, t1, label="T zone 1", color="b")

    ax5.plot(
        T1_heider_detailed[:, 0],
        T1_heider_detailed[:, 1],
        label="Zone 1 validation",
        color="g",
        lw=1,
        marker="o",
    )

    # take only every tenth data point
    no = no[::50]
    dnodca = dnodca[::50]
    p = p[::50]
    t = t[::50]
    t1 = t1[::50]
    t2 = t2[::50]
    phi_z1 = phi_z1[::50]
    phi = phi[::50]
    phi_hp = phi_hp[::50]


    no_data_val = np.hstack((np.transpose(np.atleast_2d(no_heider[:, 1])),
                         np.transpose(np.atleast_2d(no_heider[:,0])) ) )

    no_data_sim = np.hstack((np.transpose(np.atleast_2d(no)), np.transpose(np.atleast_2d(phi_z1 * 180 / np.pi) ) ))

    dno_data_val = np.hstack((np.transpose(np.atleast_2d(dnodca_heider[:, 1])),
                         np.transpose(np.atleast_2d(dnodca_heider[:,0])) ) )

    dno_data_sim = np.hstack((np.transpose(np.atleast_2d(dnodca)), np.transpose(np.atleast_2d(phi_z1 * 180 / np.pi) ) ))

    p_data_val = np.hstack((np.transpose(np.atleast_2d(p_heider[:, 1])),
                         np.transpose(np.atleast_2d(p_heider[:,0])) ) )

    p_data_sim = np.hstack((np.transpose(np.atleast_2d(p))*1e-5, np.transpose(np.atleast_2d(phi * 180 / np.pi) ) ))



    T_0_sim = np.hstack((np.transpose(np.atleast_2d(t)), np.transpose(np.atleast_2d(phi * 180 / np.pi) ) ))

    T_0_val = np.hstack((np.transpose(np.atleast_2d(T0_heider[:, 1])),
                         np.transpose(np.atleast_2d(T0_heider[:,0])) ) )

    T_1_val = np.hstack((np.transpose(np.atleast_2d(T1_heider[:, 1])),
                         np.transpose(np.atleast_2d(T1_heider[:,0])) ) )

    T_2_val = np.hstack((np.transpose(np.atleast_2d(T2_heider[:, 1])),
                         np.transpose(np.atleast_2d(T2_heider[:,0])) ) )

    T_zone_sim = np.hstack((np.transpose(np.atleast_2d(t1)), np.transpose(np.atleast_2d(t2)),
                            np.atleast_2d(phi_hp * 180 / np.pi)))

    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/NO_val.dat", no_data_val, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/NO_sim.dat", no_data_sim, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/dNO_val.dat", dno_data_val, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/dNO_sim.dat", dno_data_sim, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/p_val.dat", p_data_val, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/p_sim.dat", p_data_sim, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/T0_sim.dat", T_0_sim, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/T_1_2_sim.dat", T_zone_sim, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/T0_val.dat", T_0_val, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/T_1_val.dat", T_1_val, fmt="%.5f")
    np.savetxt("./piston_engine/validation_output_data/Heider_new_output/T_2_val.dat", T_2_val, fmt="%.5f")


    plt.show()

    return