import matplotlib.pyplot as plt
import numpy as np


def friction_kaiser(d, s, v_mean, p_in, cr):
    # Crankshaft losses
    # NOTES: model sensitive to main bearing diameter predictions

    rpm = v_mean / (2 * s) * 60

    n_b = 2  # number of bearings: (1 + n_cyl for inline and 1 + 0.5*n_cyl for V)
    d_b = (
        0.62 * d
    )  # main bearing diameter [m]. ( this is for 2 connecting rods per crank pin (typical v8))
    l_b = (
        0.4 * d_b
    )  # main bearing length [m] ( this is for 2 connecting rods per crank pin (typical v8))
    n_c = 1  # number of cylinders per crankshaft
    c_b = 0.202e6  # correlation Pa min^0.6 /m
    c_s = 9.36e1  # correlation constant Pa m^2

    fmep_bear_crank = (
        c_b * n_b * (rpm**0.6) * d_b**3 * l_b / (n_c * d**2 * s)
    )  # bearing friction
    fmeb_seal_crank = c_s * d_b / (n_c * s * d**2)  # sealing friction

    fmep_crankshaft = fmeb_seal_crank + fmep_bear_crank  # Total crankshaft losses

    # TODO: Check what b means in Keisers report
    # Piston losses
    p_a = 1e5  # ambient pressure (needs to be modified for in flight conditions)
    c_ps = 0.294e3
    c_pr = 4.06e1
    c_g = 6.89e3
    K = 0.0238  # KOLLA UPP DENNA ENHETEN
    c_bc = 3.03e2
    d_b_cr = 0.57 * d
    l_b_cr = 0.39 * d
    n_b_cr = 1  # number of connecting rod bearings (same as number of cylinders?)

    fmep_skirt = c_ps * v_mean / d  # skirt losses (says b)
    fmep_rings = c_pr * (1 + 1000 / rpm) / (d**2)  # once again b
    fmep_gas = c_g * (p_in / p_a) * (0.088 * cr + 0.182 * cr ** (1.33 - 2 * K * v_mean))
    fmep_bear_conrod = c_bc * n_b_cr * rpm * d_b_cr**3 * l_b_cr / (n_c * d**2 * s)

    fmep_piston = (
        fmep_skirt + fmep_rings + fmep_gas + fmep_bear_conrod
    )  # total piston friction losses

    # Valvetrain losses
    n_c = 2  # number of chamshaft bearings
    n_v = 4  # total number of valves

    # fmep_camshaft = c_c * rpm**0.6 * n_cs / (n_c * d**2 * s)
    # fmep_seal_camshaft = 1.2e3
    # fmep_valve = c_ff * (2 + 10 / (5 + 10))

    # fmep_valvetrain = fmep_camshaft + fmep_seal_camshaft + fmep_valve

    fmep = fmep_crankshaft + fmep_piston

    return fmep


def friction_patton(d_meter, rpm, s_meter, v_mean, p_in, cr, n_cyl, lv_max, cycle):
    # Note that this model is for spark ignition engines
    # All units here are in mm and kPa. Velocity in m/s
    # Converting all lengths to mm from m
    d = d_meter * 1000
    s = s_meter * 1000

    # OBS!!! LÄS PÅ OM HUR MAN GÖR FÖR MÅNGA CYLINDRAR
    if n_cyl == 1:
        # this is for validation case of hydrogen. this is not a vee engine
        v_engine = False
    else:
        # more than one cylinder (typical application case), we assume vee-configuration
        v_engine = True

    # Crankshaft losses
    # NOTES: model sensitive to main bearing diameter predictions
    if v_engine:
        n_b = (
            1 + 0.5 * n_cyl
        )  # number of bearings: (1 + n_cyl for inline and 1 + 0.5*n_cyl for V)
        d_b = (
            0.62 * d
        )  # main bearing diameter [mm]. ( this is for 2 connecting rods per crank pin (typical v8))
        l_b = (
            0.4 * d_b
        )  # main bearing length [mm] ( this is for 2 connecting rods per crank pin (typical v8))
    else:
        # two main bearings for single cylinder engine
        n_b = 2
        d_b = 0.6 * d  # for line engine
        l_b = 0.37 * d_b  # line engine

    n_c = n_cyl  # number of cylinders per crankshaft

    c_s = 1.22e5  # correlation constant kPa mm^2
    c_b = 3.03e-4  # correlation kPa-min/rev-mm
    c_t = 1.35e-10  # kPa-mm^2

    fmeb_seal_crank = c_s * d_b / (n_c * s * d**2)  # sealing friction
    fmep_bear_crank = (
        c_b * rpm * d_b**3 * l_b * n_b / (d**2 * s * n_c)
    )  # main bearing hydrodynamic friction
    fmep_turbulent_crank = (
        c_t * d_b**2 * rpm**2 * n_b / n_c
    )  # turbulent dissipation in the oil

    fmep_crankshaft = (
        fmeb_seal_crank + fmep_bear_crank + fmep_turbulent_crank
    )  # Total crankshaft losses

    # Piston losses
    p_a = 1e5  # atmospheric pressure
    c_ps = 2.94e2  # kPa-mm-s/m
    c_pr = 4.06e4  # kPa-mm^2
    c_g = 6.89
    K = 2.38e-2  # s/m
    c_bc = 3.03e-4

    if v_engine:
        d_b_cr = 0.57 * d  # also for 2 rods/pin V engines
        l_b_cr = 0.39 * d_b_cr  # also for 2 rods/pin V engines
    else:
        d_b_cr = 0.57 * d  # line engine
        l_b_cr = 0.41 * d_b_cr  # line engine

    n_b_cr = n_cyl  # number of connecting rod bearings (same as number of cylinders?)

    fmep_skirt = c_ps * v_mean / d  # skirt losses
    fmep_rings = c_pr * (1 + 1000 / rpm) / (d**2)  # piston rings without gas loading
    fmep_gas = (
        c_g * (p_in / p_a) * (0.088 * cr + 0.182 * cr ** (1.33 - K * v_mean))
    )  # increase in piston ring friction from gas pressure
    fmep_bear_conrod = (
        c_bc * rpm * n_b_cr * (d_b_cr**3) * l_b_cr / (n_c * (d**2) * s)
    )  # connecting rod

    fmep_piston = (
        fmep_skirt + fmep_rings + fmep_gas + fmep_bear_conrod
    )  # total piston friction losses

    # Valvetrain losses  OBS: CHOSE ONE VALVE TRAIN AND MOTIVATE (DOHC direct acting)

    if v_engine:
        camshafts = 2
    else:
        camshafts = 1

    n_cb = (
        n_b * camshafts
    )  # number of camshaft bearings ( equal number of main bearings x number of camshafts)
    n_v = 4 * n_c  # total number of valves (per cylinder or total?)

    c_cb = 2.44e2  # kPa-mm^3-min/rev
    c_cff = 133
    c_cfr = 0.005
    c_oh = 0.5
    c_om = 10.7

    fmep_camshaft = c_cb * rpm * n_cb / (n_c * (d**2) * s)
    fmep_seal_camshaft = 4.12  # camshaft bearing seals constant friction
    fmep_camfollower_flat = (
        c_cff * (1 + 1000 / rpm) * n_v / (s * n_c)
    )  # for flat cam follower
    fmep_camfollower_roller = c_cfr * rpm * n_v / (s * n_c)  # for roller cam follower.
    fmep_oscillhydro = (
        c_oh * (lv_max**1.5) * (rpm**0.5) * n_v / (d * s * n_c)
    )  # oscillation losses
    fmep_oscillmixed = c_om * (1 + 1000 / rpm) * lv_max * n_v / (s * n_c)

    fmep_valvetrain = (
        fmep_camshaft
        + fmep_seal_camshaft
        + fmep_camfollower_roller
        + fmep_oscillhydro
        + fmep_oscillmixed
    )

    # Auxiliary component losses (oil pump, water pump, non-charging alternator friction)
    fmep_aux = 6.23 + 5.22e-3 * rpm - 1.79e-7 * rpm**2

    fmep = (fmep_crankshaft + fmep_piston + fmep_valvetrain) + fmep_aux
    fmep_pe_loss = (
        fmep_crankshaft + fmep_piston + fmep_valvetrain
    )  # 0.8 Kaiser's factor
    # fmep = (fmep_crankshaft + fmep_piston) * 1e3  # for two-stroke

    # multiply by 1000 so that the unit is Pa for output
    fmep = (fmep * 1e3,)
    fmep_aux = fmep_aux * 1e3
    fmep_pe_loss = fmep_pe_loss * 1e3

    # convert to power
    if cycle == "4T":
        n_r = 2
    else:
        n_r = 1

    rps = rpm / 60

    Vd_tot = (d_meter / 2) ** 2 * np.pi * s_meter * n_cyl

    # convert losses to power
    friction_loss = (
        fmep_pe_loss * Vd_tot * rps / n_r
    )  # friction losses for total engine all cylinders
    aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses
    total_loss = friction_loss + aux_loss

    return friction_loss, aux_loss, total_loss


def scavenging(equ, phi, phi_close_out, phi_open_out, far_s, m_in_IP, rho_in, V_d, m):
    equ_cmp = equ[-1][np.argwhere(phi > phi_close_out)[0]][0]
    equ_evo = equ[-1][np.argwhere(phi > phi_open_out)[0]][0]
    m_evo = m[-1][np.argwhere(phi > phi_open_out)[0]][0]
    mr_ba = (equ_cmp / (1 + equ_evo * far_s)) * m_evo  # burned air in residual
    mr_fuel = mr_ba * far_s  # burned fuel in residual
    mr_ua = (1 / equ_evo - 1) * mr_ba  # unburned air in residual
    m_FC = (1 / equ_cmp - 1 / equ_evo) * mr_ba  # fresh charge
    m_fuelnew = (equ_evo / equ_cmp - 1) * far_s * mr_ba  # new fuel

    purity = (m_FC + mr_ua) / (m_FC + mr_ua + mr_ba + mr_fuel)
    residual_fraction = (mr_ba + mr_fuel + mr_ua) / (m_FC + mr_ba + mr_fuel + mr_ua)
    eta_trapping = m_FC / (m_in_IP[-1][-1])
    eta_charging = m_FC / (rho_in * V_d)
    # I DONT KNOW WHY THIS IS A PROBLEM FOR THE CHALMERS H2 ENGINE
    if eta_trapping == 0:
        delivery_ratio = 1.0
    else:
        delivery_ratio = eta_charging / eta_trapping
    eta_sc = 1 / (1 + (1 / purity - 1) / equ_evo)  # scavenging efficency

    return purity, residual_fraction, eta_trapping, eta_charging, delivery_ratio, eta_sc


def validation_error(phi, P, T, m, equ):
    # from sklearn.metrics import mean_squared_error
    import math

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

    mask = np.searchsorted(phi_order, ca_NASA_order * np.pi / 180)
    p_filtered = p_order[mask]
    T_filtered = T_order[mask]
    m_filtered = m_order[mask]
    equ_filtered = equ_order[mask]

    plt.scatter(phi_order[mask] * 180 / np.pi, p_filtered * 1e-5)
    plt.scatter(ca_NASA_order, p_NASA_order)
    plt.show()

    plt.scatter(phi_order[mask] * 180 / np.pi, T_filtered)
    plt.scatter(ca_NASA_order, T_NASA_order)
    plt.show()

    plt.scatter(phi_order[mask] * 180 / np.pi, m_filtered)
    plt.scatter(ca_NASA_order, m_NASA_order)
    plt.show()

    plt.scatter(phi_order[mask] * 180 / np.pi, equ_filtered)
    plt.scatter(ca_NASA_order, equ_NASA_order)
    plt.show()

    MSE_p = np.square(np.subtract(p_filtered * 1e-5, p_NASA_order)).mean()
    MSE_T = np.square(np.subtract(T_filtered, T_NASA_order)).mean()
    MSE_m = np.square(np.subtract(m_filtered, m_NASA_order)).mean()
    MSE_equ = np.square(np.subtract(equ_filtered, equ_NASA_order)).mean()

    RMSE_p = math.sqrt(MSE_p)
    RMSE_T = math.sqrt(MSE_T)
    RMSE_m = math.sqrt(MSE_m)
    RMSE_equ = math.sqrt(MSE_equ)
    print("Root Mean Square Error pressure:\n")
    print(RMSE_p)
    print("Root Mean Square Error temperature:\n")
    print(RMSE_T)
    print("Root Mean Square Error mass:\n")
    print(RMSE_m)
    print("Root Mean Square Error equivalence:\n")
    print(RMSE_equ)

    RMSE_p_rel = RMSE_p / (p_order.max() - p_order.min()) * 1e5
    RMSE_T_rel = RMSE_T / (T_order.max() - T_order.min())
    RMSE_m_rel = RMSE_m / (m_order.max() - m_order.min())
    RMSE_equ_rel = RMSE_equ / (equ_order.max() - equ_order.min())

    print("Relative RMSE pressure:\n")
    print(RMSE_p_rel)
    print("Relative RMSE temperature:\n")
    print(RMSE_T_rel)
    print("Relative RMSE mass:\n")
    print(RMSE_m_rel)
    print("Relative RMSE equ:\n")
    print(RMSE_equ_rel)

    return
