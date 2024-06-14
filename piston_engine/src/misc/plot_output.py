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

    ax1.set_ylabel(r'p [bar]')
    ax1.set_xticklabels([])
    ax1.grid()

    ax2.set_ylabel(r'T [K]')
    ax2.set_xticklabels([])
    ax2.grid()

    ax3.set_ylabel(r'V [L]')
    ax3.set_xticklabels([])
    ax3.grid()

    ax4.set_ylabel(r'rho [kg/m3]')
    ax4.set_xticklabels([])
    ax4.grid()

    ax5.set_ylabel(r'Qf [J/kg]')
    ax5.set_xticklabels([])
    ax5.grid()

    ax6.set_xlabel('phi [rad]')
    ax6.set_ylabel(r'm [kg]')
    ax6.grid()
    return


def plot_diagrams(V, P, S, T):
    fig, ax1 = plt.subplots()
    ax1.plot(V[-1] * 1000, P[-1] * 1e-5, label='last cycle')
    ax1.set_xlabel('V [L]')
    ax1.set_ylabel(r'P [bar]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax1.grid()

    fig, ax2 = plt.subplots()
    ax2.plot(S[-1], T[-1], label='last cycle')
    ax2.set_xlabel('S [J/kg]')
    ax2.set_ylabel(r'T [K]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax2.grid()
    return


def plot_essentials(phi, T, P, m, equ, validation=False):
    if validation:
        # loading validation data
        from piston_engine.src.misc.NASAdata import load_NASA
        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = load_NASA()

    fig, ax3 = plt.subplots()
    ax3.plot(phi * 180 / np.pi, T[-1], label='Our simulation')
    if validation:
        ax3.scatter(ca_NASA, T_NASA, marker="x", label='NASA-validation')
    ax3.set_xlabel('phi [deg]')
    ax3.set_ylabel(r'T [K]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax3.grid()

    fig, ax4 = plt.subplots()
    ax4.plot(phi * 180 / np.pi, P[-1] * 1e-5, label='Our simulation')
    if validation:
        ax4.scatter(ca_NASA, p_NASA, marker="x", label='NASA-validation')
    ax4.set_xlabel('phi [deg]')
    ax4.set_ylabel(r'p [bar]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax4.grid()
    # ax4.set_xlim(100,250)
    # ax4.set_ylim(5,15)

    # %%
    fig, ax5 = plt.subplots()
    ax5.plot(phi * 180 / np.pi, m[-1], label='Our simulation')
    if validation:
        ax5.scatter(ca_NASA, m_NASA, marker="x", label='NASA-validation')
    ax5.set_xlabel('phi [deg]')
    ax5.set_ylabel(r'm [kg]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax5.grid()

    fig, ax10 = plt.subplots()
    ax10.plot(phi * 180 / np.pi, equ[-1], label='Our simulation')
    if validation:
        ax10.scatter(ca_NASA, phi_NASA, marker="x", label='NASA-validation')
    ax10.set_xlabel('phi [deg]')
    ax10.set_ylabel(r'equivalence ratio [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax10.grid()

    plt.show()
    return


def plot_energy(phi, W, Q, m_in, Q_in, mf):
    fig, ax6 = plt.subplots()
    ax6.plot(phi * 180 / np.pi, W[-1], label='last cycle')
    ax6.set_xlabel('phi [deg]')
    ax6.set_ylabel(r'integral(p*dVdphi) [J]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax6.grid()

    fig, ax7 = plt.subplots()
    ax7.plot(phi * 180 / np.pi, Q[-1], label='last cycle')
    ax7.set_xlabel('phi [deg]')
    ax7.set_ylabel(r'Q [J]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax7.grid()

    fig, ax9 = plt.subplots()
    ax9.plot(phi * 180 / np.pi, Q_in[-1], label='last cycle')
    ax9.set_xlabel('phi [deg]')
    ax9.set_ylabel(r'Q_in [kg]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax9.grid()
    ax9.set_xlim([340, 400])
    plt.show()
    return


def plot_massflows(phi, m_in, mf, mdotin, mdotout, V, validation=False):
    if validation:
        from piston_engine.src.misc.NASAdata import load_NASA
        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = load_NASA()

    fig, ax8 = plt.subplots()
    ax8.plot(phi * 180 / np.pi, m_in[-1], label='last cycle')
    ax8.set_xlabel('phi [deg]')
    ax8.set_ylabel(r'm_in [kg]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax8.grid()

    fig, ax10 = plt.subplots()
    ax10.plot(phi * 180 / np.pi, mf[-1] * 1e3, label='last cycle')
    ax10.set_xlabel('phi [deg]')
    ax10.set_ylabel(r'mf [g]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax10.grid()
    ax10.set_xlim([340, 400])

    fig, ax17 = plt.subplots()
    ax17.plot(phi * 180 / np.pi, mdotin[-1], label='our simulation')
    if validation:
        ax17.scatter(ca_NASA, mdotin_NASA, marker="x", label='NASA-validation')
    ax17.set_xlabel('phi [deg]')
    ax17.set_ylabel(r'mdotin [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax17.grid()

    fig, ax18 = plt.subplots()
    ax18.plot(phi * 180 / np.pi, mdotout[-1], label='our simulation')
    if validation:
        ax18.scatter(ca_NASA, mdotout_NASA, marker="x", label='NASA-validation')
    ax18.set_xlabel('phi [deg]')
    ax18.set_ylabel(r'mdotout [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax18.grid()

    fig, ax19 = plt.subplots()
    ax19.plot(phi * 180 / np.pi, V[-1] * 1000, label='our simulation')
    ax19.set_xlabel('phi [deg]')
    ax19.set_ylabel(r'Vt [l]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax19.grid()

    plt.show()
    return


def plot_manifolds(phi, equ_IP, m_IP, T_IP, equ_EP, m_EP, T_EP):
    fig, ax11 = plt.subplots()
    ax11.plot(phi * 180 / np.pi, equ_IP[-1], label='last cycle')
    ax11.set_xlabel('phi [deg]')
    ax11.set_ylabel(r'equivalence ratio IP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax11.grid()

    fig, ax12 = plt.subplots()
    ax12.plot(phi * 180 / np.pi, m_IP[-1], label='last cycle')
    ax12.set_xlabel('phi [deg]')
    ax12.set_ylabel(r'mass IP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax12.grid()

    fig, ax13 = plt.subplots()
    ax13.plot(phi * 180 / np.pi, T_IP[-1], label='last cycle')
    ax13.set_xlabel('phi [deg]')
    ax13.set_ylabel(r'T IP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax13.grid()

    fig, ax14 = plt.subplots()
    ax14.plot(phi * 180 / np.pi, equ_EP[-1], label='last cycle')
    ax14.set_xlabel('phi [deg]')
    ax14.set_ylabel(r'equivalence ratio EP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax14.grid()

    fig, ax15 = plt.subplots()
    ax15.plot(phi * 180 / np.pi, m_EP[-1], label='last cycle')
    ax15.set_xlabel('phi [deg]')
    ax15.set_ylabel(r'mass EP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax15.grid()

    fig, ax16 = plt.subplots()
    ax16.plot(phi * 180 / np.pi, T_EP[-1], label='last cycle')
    ax16.set_xlabel('phi [deg]')
    ax16.set_ylabel(r'T EP [-]')
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax16.grid()
    return


def plot_convergence(pdiff, Tdiff, mdiff, equdiff, T_out_diff):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

    ax1.plot(pdiff[1:])
    ax2.plot(Tdiff[1:])
    ax3.plot(mdiff[1:])
    ax4.plot(equdiff[1:])
    ax5.plot(T_out_diff[1:])

    ax1.set_ylabel(r'pdiff [Pa]')
    ax1.set_xticklabels([])
    ax1.grid()
    ax1.set_yscale('log')

    ax2.set_ylabel(r'Tdiff [K]')
    ax2.set_xticklabels([])
    ax2.grid()
    ax2.set_yscale('log')

    ax3.set_ylabel(r'mdiff [mg]')
    ax3.set_xticklabels([])
    ax3.grid()
    ax3.set_yscale('log')

    ax4.set_ylabel(r'equdiff [-]')
    ax4.set_xticklabels([])
    ax4.grid()
    ax4.set_yscale('log')

    ax5.set_ylabel(r'T_out_diff [-]')
    ax5.set_xticklabels([])
    ax5.grid()
    ax5.set_yscale('log')

    plt.show()
    return


def plot_details(phi, P, T, phi_open_in, phi_close_in, phi_open_out, phi_close_out, phi_sc,  validation=False):
    p_order = np.roll(P[-1], -np.argwhere(phi * 180 / np.pi > 180)[0][0])
    T_order = np.roll(T[-1], -np.argwhere(phi * 180 / np.pi > 180)[0][0])
    phi_order = phi - phi[0]
    if validation:
        from piston_engine.src.misc.NASAdata import load_NASA
        ca_NASA, p_NASA, T_NASA, m_NASA, phi_NASA, mdotin_NASA, mdotout_NASA = load_NASA()
        p_NASA_order = np.roll(p_NASA, -26)
        T_NASA_order = np.roll(T_NASA, -26)
        ca_NASA_order = ca_NASA - 50

    lw_v = 3
    figsize = (16, 5)

    fig, ax3 = plt.subplots(figsize=figsize)
    ax3.plot(phi_order * 180 / np.pi, T_order, label='Simulation', color="k", lw=2)
    if validation:
        ax3.scatter(ca_NASA_order, T_NASA_order, marker="X", label='NASA-validation', color="k", s=64)
    ax3.axvline(x=phi_open_out * 180 / np.pi + 180, color='b', ls='--', lw=lw_v)
    ax3.axvline(x=phi_open_in * 180 / np.pi + 180, color='r', ls='--', lw=lw_v)
    ax3.axvline(x=phi_close_out * 180 / np.pi - 180, color='g', ls='--', lw=lw_v)
    ax3.axvline(x=phi_close_in * 180 / np.pi - 180, color='k', ls='--', lw=lw_v)
    ax3.axvline(x=phi_sc * 180 / np.pi - 180, color='m', ls='--', lw=lw_v)
    ax3.text(phi_open_out * 180 / np.pi + 180 + 3, 2250, "EVO", fontsize=24)
    ax3.text(phi_open_in * 180 / np.pi + 180 + 3, 2250, "IVO", fontsize=24)
    ax3.text(phi_close_out * 180 / np.pi - 180 + 3, 2250, "EVC", fontsize=24)
    ax3.text(phi_close_in * 180 / np.pi - 180 + 3, 2250, "IVC", fontsize=24)
    ax3.text(phi_sc * 180 / np.pi - 180 + 3, 2250, "SOC", fontsize=24)
    ax3.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=32)
    ax3.set_ylabel(r'Temperature $T$ [K]', fontsize=32)
    ax3.set_title(r'$T - \theta$ diagram', fontsize=32)
    ax3.set_xlim(45, 405)
    ax3.set_xticks([45, 90, 135, 180, 225, 270, 315, 360, 405])
    ax3.tick_params(labelsize=24)
    plt.legend(loc='best', frameon=True, fontsize=24)
    ax3.grid()

    fig, ax4 = plt.subplots(figsize=figsize)
    ax4.plot(phi_order * 180 / np.pi, p_order * 1e-5, label='Simulation', color="k", lw=2)
    if validation:
        ax4.scatter(ca_NASA_order, p_NASA_order, marker="X", label='NASA-validation', color="k", s=64)
    ax4.axvline(x=phi_open_out * 180 / np.pi + 180, color='b', ls='--', lw=lw_v)
    ax4.axvline(x=phi_open_in * 180 / np.pi + 180, color='r', ls='--', lw=lw_v)
    ax4.axvline(x=phi_close_out * 180 / np.pi - 180, color='g', ls='--', lw=lw_v)
    ax4.axvline(x=phi_close_in * 180 / np.pi - 180, color='k', ls='--', lw=lw_v)
    ax4.axvline(x=phi_sc * 180 / np.pi - 180, color='m', ls='--', lw=lw_v)
    ax4.text(phi_open_out * 180 / np.pi + 180 + 3, 175, "EVO", fontsize=24)
    ax4.text(phi_open_in * 180 / np.pi + 180 + 3, 175, "IVO", fontsize=24)
    ax4.text(phi_close_out * 180 / np.pi - 180 + 3, 175, "EVC", fontsize=24)
    ax4.text(phi_close_in * 180 / np.pi - 180 + 3, 175, "IVC", fontsize=24)
    ax4.text(phi_sc * 180 / np.pi - 180 + 3, 2250, "SOC", fontsize=24)
    ax4.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=32)
    ax4.set_ylabel(r'Pressure $p$ [bar]', fontsize=32)
    ax4.set_title(r'$p - \theta$ diagram', fontsize=32)
    ax4.set_xlim(45, 405)
    ax4.set_xticks([45, 90, 135, 180, 225, 270, 315, 360, 405])
    ax4.tick_params(labelsize=24)
    plt.legend(loc='best', frameon=True, fontsize=24)
    ax4.grid()

    plt.show()
    return


def plot_validation(phi, P, T, m, equ):
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
    ax1.plot(phi_order * 180 / np.pi, m_order*1000, label='Piston model', color="k", lw=4)
    ax1.scatter(ca_NASA_order, m_NASA_order*1000, marker="X", label='NASA-CR-185155', color="r", s=512)
    ax1.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
    ax1.set_ylabel(r'Mass $m$ [g]', fontsize=fs)
    ax1.set_title(r'$m - \theta$ diagram', fontsize=fs)
    ax1.set_xlim(0, 360)
    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.tick_params(labelsize=fs)
    plt.legend(loc='best', frameon=True, fontsize=fs)
    ax1.grid()
    plt.savefig('simulation_data/figures/m_validation.pdf', dpi=res, bbox_inches='tight')

    fig, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(phi_order * 180 / np.pi, equ_order, label='Piston model', color="k", lw=4)
    ax2.scatter(ca_NASA_order, equ_NASA_order, marker="X", label='NASA-CR-185155', color="r", s=512)
    ax2.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
    ax2.set_ylabel(r'Equivalence ratio $\phi$ [-]', fontsize=fs)
    ax2.set_title(r'$\phi - \theta$ diagram', fontsize=fs)
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.tick_params(labelsize=fs)
    plt.legend(loc='best', frameon=True, fontsize=fs)
    ax2.grid()
    plt.savefig('simulation_data/figures/equ_validation.pdf', dpi=res, bbox_inches='tight')

    fig, ax3 = plt.subplots(figsize=figsize)
    ax3.plot(phi_order * 180 / np.pi, T_order, label='Piston model', color="k", lw=4)
    ax3.scatter(ca_NASA_order, T_NASA_order, marker="X", label='NASA-CR-185155', color="r", s=512)
    ax3.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
    ax3.set_ylabel(r'Temperature $T$ [K]', fontsize=fs)
    ax3.set_title(r'$T - \theta$ diagram', fontsize=fs)
    ax3.set_xlim(0, 360)
    ax3.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax3.tick_params(labelsize=fs)
    plt.legend(loc='best', frameon=True, fontsize=fs)
    ax3.grid()
    plt.savefig('simulation_data/figures/T_validation.pdf', dpi=res, bbox_inches='tight')

    fig, ax4 = plt.subplots(figsize=figsize)
    ax4.plot(phi_order * 180 / np.pi, p_order * 1e-5, label='Piston model', color="k", lw=4)
    ax4.scatter(ca_NASA_order, p_NASA_order, marker="X", label='NASA-CR-185155', color="r", s=512)
    ax4.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
    ax4.set_ylabel(r'Pressure $p$ [bar]', fontsize=fs)
    ax4.set_title(r'$p - \theta$ diagram', fontsize=fs)
    ax4.set_xlim(0, 360)
    ax4.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax4.tick_params(labelsize=fs)
    plt.legend(loc='best', frameon=True, fontsize=fs)
    ax4.grid()
    plt.savefig('simulation_data/figures/p_validation.pdf', dpi=res, bbox_inches='tight')
    plt.show()

    # create output data for tikz
    phi_transpose = np.atleast_2d(phi_order * 180 / np.pi).T
    phi_transpose = phi_transpose[::100]

    m_transpose = np.atleast_2d(m_order*1000).T
    m_transpose = m_transpose[::100]

    equ_transpose = np.atleast_2d(equ_order).T
    equ_transpose = equ_transpose[::100]

    T_transpose = np.atleast_2d(T_order).T
    T_transpose = T_transpose[::100]

    p_transpose = np.atleast_2d(p_order * 1e-5).T
    p_transpose = p_transpose[::100]



    ca_NASA_transpose = np.atleast_2d(ca_NASA_order).T
    m_NASA_transpose = np.atleast_2d(m_NASA_order*1000).T
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


    np.savetxt("validation_output_data/m_validation.dat", m_true, fmt='%.5f')
    np.savetxt("validation_output_data/m_simulation.dat", m_sim, fmt='%.5f')

    np.savetxt("validation_output_data/equ_validation.dat", equ_true, fmt='%.5f')
    np.savetxt("validation_output_data/equ_simulation.dat", equ_sim, fmt='%.5f')

    np.savetxt("validation_output_data/T_validation.dat", T_true, fmt='%.5f')
    np.savetxt("validation_output_data/T_simulation.dat", T_sim, fmt='%.5f')

    np.savetxt("validation_output_data/p_validation.dat", p_true, fmt='%.5f')
    np.savetxt("validation_output_data/p_simulation.dat", p_sim, fmt='%.5f')

    return


def plot_rohr(phi, Q, Q_in, v, apiston, dtdphi, bore, p, t):
    fs = 24
    fig, ax7 = plt.subplots()
    ax7.plot(phi * 180 / np.pi, Q, label='last cycle')
    ax7.set_xlabel('phi [deg]')
    ax7.set_ylabel(r'Q [J]')
    ax7.set_title(r'heat loss, integrated', fontsize=fs)
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax7.grid()

    fig, ax9 = plt.subplots()
    ax9.plot(phi * 180 / np.pi, Q_in, label='last cycle')
    ax9.set_xlabel('phi [deg]')
    ax9.set_ylabel(r'Q_in [J]')
    ax9.set_title(r'heat addition, integrated', fontsize=fs)
    plt.legend(loc='best', fontsize='small', frameon=False)
    ax9.grid()
    ax9.set_xlim([340, 400])


    phi_deg = phi * 180 / np.pi
    rohr = np.diff(Q_in) / np.diff(phi_deg)
    phi2 = (phi_deg[:-1] + phi_deg[1:]) / 2

    fig, ax10 = plt.subplots()
    ax10.plot(phi2, rohr)
    ax10.set_xlabel('phi [deg]')
    ax10.set_ylabel(r'ROHR [J/CA]')
    ax10.set_title(r'ROHR', fontsize=fs)
    ax10.set_xlim(310, 410)
    ax10.set_ylim(0, 150)
    ax10.set_xticks([310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410])
    ax10.grid()

    area = (v / apiston) * bore * np.pi + apiston * 2  # total cylinder area
    area2 = (area[:-1] + area[1:]) / 2  # avg just to plot

    wall_heatloss_rad = np.diff(Q) / np.diff(phi)  # Joule per radian
    wall_heatloss = wall_heatloss_rad / dtdphi  # Joule per second (Watt)
    wall_heatflux = wall_heatloss / area2  # Watt / m^2
    wall_heatflux = wall_heatflux * 1e-4   # Watt / cm^2

    twall = 500  # wall temperature
    t2 = (t[:-1] + t[1:]) / 2  # avg temperature to fit array length
    alpha = wall_heatflux * 1e4 / (t2 - twall)  # W/(m^2 K) heat transfer coefficient

    phi_deg2 = (phi_deg[:-1] + phi_deg[1:]) / 2

    fig, ax11 = plt.subplots()
    ax11.plot(phi2, wall_heatflux)
    ax11.set_xlabel('phi [deg]')
    ax11.set_ylabel(r'Wall heat flux [W/cm2]')
    ax11.set_title(r'Wall heat flux', fontsize=fs)
    ax11.set_xlim(320, 420)
    ax11.set_xticks([320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420])
    ax11.grid()

    fig, ax12 = plt.subplots()
    ax12.plot(phi_deg, p*1e-5)
    ax12.set_xlabel('phi [deg]')
    ax12.set_ylabel(r'Cylinder pressure [bar]')
    ax12.set_title(r'Pressure', fontsize=fs)
    ax12.set_xlim(310, 410)
    ax12.set_ylim(0, 60)
    ax12.set_xticks([310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410])
    ax12.grid()

    fig, ax13 = plt.subplots()
    ax13.plot(phi_deg2, alpha)
    ax13.set_xlabel('phi [deg]')
    ax13.set_ylabel(r'Heat transfer coefficient [W/ (m^2 K)]')
    ax13.set_title(r'Heat transfer coefficient', fontsize=fs)
    ax13.set_xlim(320, 420)
    #ax13.set_ylim(0, 60)
    ax13.set_xticks([320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420])
    ax13.grid()
    plt.show()

    # saving heat transfer coefficient to ba able to validate
    #np.savetxt("simulation_data/htc.csv", alpha, delimiter=",")
    #p.savetxt("simulation_data/phi2.csv", phi_deg2, delimiter=",")
    return


def plot_pv(p, v):
    fs = 24
    fig, ax12 = plt.subplots()
    ax12.plot(v*1e3, p*1e-5)
    ax12.set_xlabel('Volume [liter]')
    ax12.set_ylabel(r'Cylinder pressure [bar]')
    ax12.set_title(r'pV-diagram', fontsize=fs)
    ax12.grid()
    plt.show()

    return
