import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

from piston_engine.src.piston import valve_isentrop, walls, wiebe, port_isentrop, twozone_model, nox_model_cantera
#from piston_engine.src.piston import thermo_computations, polynomials
from piston_engine.src.misc import post_processing
from piston_engine.src.misc.entropy import entropy_calc

#from piston_engine.src.piston.fuel_func import fuel_props
#from piston_engine.src.piston.fluid_props import properties

import thermo

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


from piston_engine.src.misc.output import output_thermo, output_power, output_efficiencies

from piston_engine.src.misc.plot_output import plot_validation


from piston_engine.src.misc.plot_output import plot_rohr, plot_pvts

from piston_engine.src.misc.plot_output import plot_energy, plot_convergence, plot_progress
from piston_engine.src.misc.plot_output import plot_essentials, plot_manifolds, plot_details, plot_massflows


from piston_engine.src.misc.output import output_thermo_validation, output_power_validation, \
    output_scavenging_validation, output_efficiencies_validation

from numba import jit

#Runiv = 8.3144626  # J mol^-1 K^-1

dPdphi_temp = 0


#@jit(nopython=True)
def run_piston_engine(indata, flags):
    [p_in, T_in, p_ratio, cycle, thermo_unused, cooling, opposed, cr, d, bsr, v_mean, lms, Twalls, ch,
     valve_timings, n_valve, lv_max, cd, eta_c, mf_tot, wa, wm, m_wiebe,
     phi_sc, phi_cd, T_fuel, p_fuel, it, wiebe_type, valve_type, far_goal, cylinders, fuel_type,
     c1, c4, c5] = indata

    p_out = p_ratio * p_in

    Twalls = np.array(Twalls)

    phi_open_in = valve_timings[0]
    phi_close_in = valve_timings[1]
    phi_open_out = valve_timings[2]
    phi_close_out = valve_timings[3]

    # number of outputs from the function
    nr_output = 16

    s = d/bsr  # stroke retrieved from bore stroke ratio
    l_con = s/(2*lms)  # connecting rod length
    rpm = 60*v_mean/(2*s)  # rpm from mean velocity

    V_d = d ** 2 * (np.pi / 4) * s  # displacement

    if opposed:
        # Opposed piston has double volume
        V_d = V_d * 2

    V_clearance = V_d / (cr - 1)
    V_max = V_clearance + V_d

    rpm = v_mean / (2 * s) * 60

    h_in, _, _, _, R_in, _, _, _ = thermo.mixture(t=T_in, p=p_in, equ=0, fuel_type=fuel_type)  # assuming pure air in intake

    #R_in = Runiv / M_in
    rho_in = p_in / (R_in * T_in)

    far_s, LHV = thermo.fuel_props(fuel_type)

    phi_ec = (phi_sc + phi_cd)  # angle at combustion end

    if fuel_type == 'jetA':
        cp_fuel, h_fuel, _, _ = thermo.polynomials.JETA(T_fuel)
    elif fuel_type == 'H2':
        cp_fuel, h_fuel, s_fuel, _, M_fuel = thermo.polynomials.H2(T_fuel, p_fuel)
    else:
        raise Exception('Unknown fuel.')

    # Four stroke/two stroke
    if cycle == "2T":
        d_port = 0.55 * d  # diameter of intake and exhaust port
        cycle_phi = 2 * np.pi
        phi_start = phi_open_out
    elif cycle == "4T":
        cycle_phi = 4 * np.pi
        phi_start = phi_close_in - cycle_phi
    else:
        raise Exception(f'Unknown cycle. The cycle was {cycle}.')

    Apiston = (np.pi / 4) * d ** 2
    Ahead = Apiston

    # Starts here (vilken välja?)
    #n = 10000
    n = 50000
    #n = 20000

    phi = np.linspace(phi_start, cycle_phi + phi_start, n)
    dtdphi = s / (np.pi * v_mean)

    # initial values, P, T, rho. Using inlet values
    P0 = p_in
    T0 = T_in

    rho0 = P0 / (R_in * T0)

    if opposed:
        # Opposed piston
        V1 = 2 * (np.pi / 4 * d ** 2 * s * (
                1 / (cr - 1) + 0.5 * (1 - np.cos(phi) + lms / 4 * (1 - np.cos(2 * phi)))))  # everything doubled
    else:
        V1 = np.pi / 4 * d ** 2 * s * (1 / (cr - 1) + 0.5 * (1 - np.cos(phi) + lms / 4 * (1 - np.cos(2 * phi))))

    if valve_type == "port":
        height_open_in = V1[np.argwhere(phi >= phi_open_in)[0][0]] / (np.pi * 0.25 * d ** 2)
        height_open_out = V1[np.argwhere(phi >= phi_open_out)[0][0]] / (np.pi * 0.25 * d ** 2)

    m0 = rho0 * V1[0]

    if 'fuel_mass' in flags:
        # this is used for efficiency calcuations
        Qf = LHV * mf_tot

        # this is only used for initial value of equivalence ratio in exhaust port
        m_air_theo = V1[np.argwhere(phi >= phi_close_out)[0][0]] * rho_in
        far_tot = mf_tot / m_air_theo
    else:
        # Calculating fuel mass based on desired fuel air ratio (based on total flow)
        # this is just an initial guess
        if cycle == "4T":
            # 4 stroke starts at inlet closing, so first value is inlet closing volume
            m_air_theo = V1[0] * rho_in
        mf_tot = far_goal * m_air_theo
        # far_tot is used for init
        far_tot = far_goal
        Qf = LHV * mf_tot  #hmmm eta_c or not

    @jit(nopython=True)
    def dxdphi(phi, x, Pref, Tref, Vref, Pmotor, Vmotor):

        # solve a system of ODEs for pressure, temperature, volume
        # assign ode to vector element

        T = x[0]  # temperature
        V = x[1]  # volume
        #Q = x[2]  # heat loss
        #Qin = x[3]  # heat in
        m = x[4]  # mass
        P = x[5]
        #m_in = x[7]
        #mf = x[8]
        equ = x[9]
        #m_out = x[10]
        #energy_out = x[11]
        m_IP = x[12]
        T_IP = x[13]
        equ_IP = x[14]
        m_EP = x[15]
        T_EP = x[16]
        equ_EP = x[17]
        # m_in_IP = x[18]
        # entropy_cyl = x[19]
        # mdotin = 20
        # mdotout = 21
        #mout_EP = x[22]
        #Q_app = x[23]  # apparent heat release


        # Gas properties inside the cylinder
        h, u, cp, cv, R, gamma, entropy, _ = thermo.mixture(T, P, equ, fuel_type)

        # Intake port values
        h_IP, u_IP, cp_IP, cv_IP, R_IP, gamma_IP, entropy_IP, _ = thermo.mixture(T_IP, p_in, equ_IP, fuel_type)

        # Exhaust port values
        h_EP, u_EP, cp_EP, cv_EP, R_EP, gamma_EP, entropy_EP, _ = thermo.mixture(T_EP, p_out, equ_EP, fuel_type)

        # Phi derivative of the volume (without Taylor expansion)
        # Opposed piston
        if opposed:
            dVdphi = 2 * (np.pi / 4) * d ** 2 * (s / 2) * (
                    np.sin(phi) + (lms / 2) * np.sin(2 * phi) / np.sqrt(1 - lms ** 2 * np.sin(phi) ** 2))
        else:
            dVdphi = (np.pi / 4) * d ** 2 * (s / 2) * (
                    np.sin(phi) + (lms / 2) * np.sin(2 * phi) / np.sqrt(1 - lms ** 2 * np.sin(phi) ** 2))

        # Wiebe func (heat release rate)
        t = phi / np.pi * s / v_mean
        t_soc = phi_sc / np.pi * s / v_mean
        t_eoc = phi_ec / np.pi * s / v_mean

        # Rate of added heat through combustion
        if wiebe_type == "Kaiser":
            dqfdphi = wiebe.dqfdt_kaiser(Qf, t, t_soc, t_eoc, wa, wm) * dtdphi
            # Rate of injected fuel mass
            dmfdphi = dqfdphi / (LHV * eta_c)
        elif wiebe_type == "Single":
            #this will integrate to 99.9% of Qf
            dqfdphi = wiebe.dqfdt_single(phi, m_wiebe, phi_sc, phi_cd, Qf)
            # Rate of injected fuel mass
            dmfdphi = dqfdphi / (LHV * eta_c)
            #dmfdphi = dqfdphi / LHV
        elif wiebe_type == "Double":
            # Rate of injected fuel mass
            dmfdphi = wiebe.dmfdphi_double(phi, c1, phi_sc, phi_cd, c4, c5, mf_tot)
            # Rate of injected heat
            dqfdphi = dmfdphi * (LHV * eta_c)
        elif wiebe_type == "Single_mass":
            # Rate of injected fuel mass
            dmfdphi = wiebe.dmfdphi_single_mass(phi, m_wiebe, phi_sc, phi_cd, mf_tot)
            # Rate of injected heat (only used for two zone calculations)
            dqfdphi = dmfdphi * LHV
        else:
            raise Exception(f'Unknown Wiebe function. The Wiebe function was {wiebe_type}.')

        # Heat loss
        ref = (Pref, Tref, Vref, Pmotor, Vmotor)
        Awall = (V / Apiston) * (d * np.pi)  # Wall area in contact with the fluid

        # removed opposed piston functionality for now
        #if opposed:
        #    Awalls = [Awall, Apiston, Apiston]
        #else:
        #    Awalls = [Awall, Apiston, Ahead]
        Awalls = np.array([Awall, Apiston, Ahead])

        if cooling == "Woschni":
            if np.mod(phi, cycle_phi) > phi_sc and np.mod(phi, cycle_phi) < phi_open_out:
                firing = True
            else:
                firing = False

            if cycle == "4T":
                if phi_close_in - cycle_phi < np.mod(phi, cycle_phi) < phi_open_out:
                    #print(phi_close_in * 180 / np.pi , phi * 180 / np.pi , phi_open_out * 180 / np.pi )
                    if fuel_type == "jetA":
                        dqdphi, alpha = walls.dqdphi(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                     "HP-process", firing)
                    else:
                        # for hydrogen
                        #if firing:
                        #    print(P*1e-5, phi*180 / np.pi)
                        dqdphi, alpha = walls.dqdphi_woschni_h2(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                     "HP-process", firing)
                else:
                    if fuel_type == "jetA":
                        dqdphi, alpha = walls.dqdphi(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                     "Charge changing", firing)
                    else:
                        # for hydrogen
                        dqdphi, alpha = walls.dqdphi_woschni_h2(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                     "Charge changing", firing)

            elif cycle == "2T":
                if phi_close_out < phi < phi_open_out:
                    dqdphi, alpha = walls.dqdphi(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                 "HP-process", firing)
                else:
                    dqdphi, alpha = walls.dqdphi(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                                 "Charge changing", firing)
            else:
                raise Exception(f'Unknown cycle. The cycle was {cycle}.')

        elif cooling == "Hohenberg":
            dqdphi = walls.dqdphi_hohenberg(T, P, V, Twalls, Awalls, v_mean, dtdphi)
        elif cooling == "H2":
            dqdphi = walls.dqdphi_h2(T, P, Twalls, Awalls, d, v_mean, ref, dqfdphi, dtdphi)
        else:
            raise Exception(f'Unknown cooling model. The cooling model was {cooling}.')

        dqdphi = ch * dqdphi

        # Continuity equation
        if cycle == "4T":
            dmindphi = valve_isentrop.dmvdphi(phi, phi_open_in, phi_close_in, n_valve, lv_max, cd, d, P, T, gamma, R,
                                              p_in, T_IP, gamma_IP, R_IP, "in", cycle_phi)
            dmoutdphi = valve_isentrop.dmvdphi(phi, phi_open_out, phi_close_out, n_valve, lv_max, cd, d, P, T, gamma, R,
                                               p_out, T_EP, gamma_EP, R_EP, "out", cycle_phi)
        elif cycle == "2T":
            height = V / (np.pi * 0.25 * d ** 2)
            dmindphi = port_isentrop.dmvdphi(height, height_open_in, n_valve, cd, d_port, P, T, gamma, R, p_in, T_IP,
                                             gamma_IP, R_IP, "in")
            dmoutdphi = port_isentrop.dmvdphi(height, height_open_out, n_valve, cd, d_port, P, T, gamma, R, p_out, T_EP,
                                              gamma_EP, R_EP, "out")
        else:
            raise Exception(f'Unknown cycle. The cycle was {cycle}.')

        dmindphi = dmindphi * dtdphi
        dmoutdphi = dmoutdphi * dtdphi

        dmdphi = dmindphi - dmoutdphi + dmfdphi

        x[20] = dmindphi
        x[21] = dmoutdphi

        # INTAKE PORT CALCULATIONS    
        if dmindphi > 0:
            h_out_IP = h_IP
            equ_out_IP = equ_IP
        else:
            h_out_IP = h
            equ_out_IP = equ

        h_in_IP = h_in

        dellRdellequ_IP, delludellequ_IP = \
            thermo.equivalence_derivative(equ_IP, T_IP, p_in, fuel_type)

        term1 = T_IP + h_out_IP / cv_IP + \
            (1 + equ_IP * far_s) * (equ_IP - equ_out_IP) / (cv_IP * (1 + equ_out_IP * far_s)) * delludellequ_IP - \
            (T_IP / R_IP) * (1 + equ_IP * far_s) * (equ_IP - equ_out_IP) / (1 + equ_out_IP * far_s)

        term2 = h_in_IP / cv_IP + T_IP + (equ_IP * (1 + equ_IP * far_s) / cv_IP) * delludellequ_IP - \
            (T_IP * equ_IP / R_IP) * (1 + equ_IP * far_s) * dellRdellequ_IP

        # mass inflow to intake port
        dmindphi_IP = dmindphi * term1 / term2

        # Continuity equation of the intake port
        dmdphi_IP = dmindphi_IP - dmindphi

        # Change of equivalence ratio in the intake port
        dequdphi_IP = ((1 + equ_IP * far_s) / m_IP) * (
                (equ_IP - equ_out_IP) * dmindphi / (1 + equ_out_IP * far_s) - equ_IP * dmindphi_IP)

        # Change of gas constant in intake port
        dRdphi_IP = dellRdellequ_IP * dequdphi_IP

        # Change of intake port temperature
        dTdphi_IP = (-R_IP * T_IP * dmdphi_IP - m_IP * T_IP * dRdphi_IP) / (R_IP * m_IP)

        # EXHAUST PORT CALCULATIONS
        if dmoutdphi > 0:
            h_in_EP = h
            equ_in_EP = equ

        else:
            h_in_EP = h_EP
            equ_in_EP = equ_EP

        # Assuming backflow into exhaust port has same properties as exhaust port
        h_out_EP = h_EP

        dellRdellequ_EP, delludellequ_EP = \
            thermo.equivalence_derivative(equ_EP, T_EP, p_out, fuel_type)

        # Change of equivalence ratio in the exhaust port
        dequdphi_EP = dmoutdphi * (1 + equ_EP * far_s) * (equ_in_EP - equ_EP) / (1 + equ_in_EP * far_s)

        # Change of gas constant in exhaust port
        dRdphi_EP = dellRdellequ_EP * dequdphi_EP

        # Mass outflow of exhaust port
        term1 = m_EP * delludellequ_EP * dequdphi_EP + dmoutdphi * (u_EP - h_in_EP - cv_EP * T_EP) - \
                (cv_EP * m_EP * T_EP / R_EP) * dRdphi_EP

        term2 = u_EP - h_out_EP - cv_EP * T_EP
        dmoutdphi_EP = term1 / term2

        # Continuity equation of the exhaust port
        dmdphi_EP = dmoutdphi - dmoutdphi_EP

        # Change of exhaust port temperature
        dTdphi_EP = (-R_EP * T_EP * dmdphi_EP - m_EP * T_EP * dRdphi_EP) / (R_EP * m_EP)

        # CYLINDER CALCULATIONS
        # Change in equivalence ratio in cylinder
        dequdphi = ((1 + equ * far_s) / m) * (((equ_out_IP - equ) / (1 + equ_out_IP * far_s)) * dmindphi
                                              - ((equ_in_EP - equ) / (1 + equ_in_EP * far_s)) * dmoutdphi
                                              + dmfdphi / far_s)

        dellRdellequ, delludellequ = thermo.equivalence_derivative(equ, T, P, fuel_type)

        dRdphi = dellRdellequ * dequdphi

        if dmindphi > 0:
            h_in_cyl = h_IP
        else:
            h_in_cyl = h


        if dmoutdphi > 0:
            h_out_cyl = h
        else:
            h_out_cyl = h_EP

        # Energy equation
        # no heat addition term here. It is incorporated in the enthalpies
        dudphi = (- dqdphi - P * dVdphi - u * dmdphi + dmfdphi * h_fuel +
                  dmindphi * h_in_cyl - dmoutdphi * h_out_cyl) / m

        dTdphi = (dudphi - delludellequ * dequdphi) / cv

        # Ideal gas law
        dPdphi = (m * R * dTdphi + R * T * dmdphi + m * T * dRdphi - P * dVdphi) / V

        # if P > 150e5:
        #    dPdphi = dPdphi * 1.05

        # Entropy (Gibbs equation)
        dsdphi = dudphi / T + dVdphi * ( P / T)

        # Apparent rate of heat release
        dQ_appdphi = (1 / (1.4 - 1)) * V * dPdphi + (1 / (1.4 - 1)) * P * dVdphi
        #dQ_appdphi = (1 / (gamma - 1)) * V * dPdphi + (1 / (gamma - 1)) * P * dVdphi

        #if np.mod(phi, 0.1) < 0.01:
        #    print(phi*180/np.pi, T, P*1e-5, dmindphi, dmoutdphi, dmfdphi, equ, m)

        return [dTdphi, dVdphi, dqdphi, dqfdphi, dmdphi, dPdphi,
                P * dVdphi, dmindphi, dmfdphi, dequdphi, dmoutdphi, h_out_EP * dmoutdphi_EP,
                dmdphi_IP, dTdphi_IP, dequdphi_IP, dmdphi_EP, dTdphi_EP, dequdphi_EP,
                dmindphi_IP, dsdphi, 0.0, 0.0, dmoutdphi_EP, dQ_appdphi, h_in_IP * dmindphi_IP, h_fuel * dmfdphi]

    # from initial guess of fuel air ratio
    equ_EP0 = far_tot / far_s

    # Init simulation
    x = [T0, V1[0], 0.0, 0.0, m0, P0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1*m0, T_in, 0.0, m0 * 1e-3, T_in, equ_EP0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # initial state value

    T = []
    V = []
    Q = []
    Q_in = []
    m = []
    P = []
    W = []
    m_in = []
    mf = []
    equ = []
    m_out = []
    energy_out = []
    m_IP = []
    T_IP = []
    equ_IP = []
    m_EP = []
    T_EP = []
    equ_EP = []
    m_in_IP = []
    S = []
    mdotin = []
    mdotout = []
    m_out_EP = []
    T_out = []
    Q_apparent = []
    energy_in = []
    fuel_enthalpy_in = []

    mdiff = []
    pdiff = []
    Tdiff = []
    equdiff = []
    T_out_diff = []
    mf_diff = []
    mEP_diff = []
    mIP_diff = []

    Pref = p_in
    Tref = T_in
    if cycle == "2T":
        Vref = V1[np.argwhere(phi >= phi_close_out)[0][0]]
    elif cycle == "4T":
        Vref = V1[0]
    Pmotor = p_in
    Vmotor = V1[np.argwhere(phi >= phi_sc)[0][0]]
    start = timer()

    for i in range(it):
        # Iterate the engine simulation until convergence criterion is met
        if i > 0:
            x = [T[-1][-1], V[-1][-1], 0, 0.0, m[-1][-1], P[-1][-1], 0.0, 0.0, 0.0,
                 equ[-1][-1], 0.0, 0.0, m_IP[-1][-1], T_IP[-1][-1], equ_IP[-1][-1],
                 m_EP[-1][-1], T_EP[-1][-1], equ_EP[-1][-1], 0, S[-1][-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if cycle == "2T":
                Pref = P[-1][np.argwhere(phi > phi_close_out)[0]][0]
                Tref = T[-1][np.argwhere(phi > phi_close_out)[0]][0]
                Vref = V[-1][np.argwhere(phi > phi_close_out)[0]][0]
            elif cycle == "4T":
                Pref = P[-1][-1]
                Tref = T[-1][-1]
                Vref = V[-1][-1]
            try:
                Pmotor = P[-1][np.argwhere(phi > phi_sc)[0]][0]
                Vmotor = V[-1][np.argwhere(phi > phi_sc)[0]][0]
            except IndexError as e:
                print(e)
                return np.zeros(nr_output)

        woschni_args = (Pref, Tref, Vref, Pmotor, Vmotor)
        if cycle == "2T":
            #sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='Radau', y0=x, t_eval=phi,
            #                rtol=1e-12)
            sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='LSODA', y0=x, t_eval=phi,
                            rtol=1e-10, atol=1e-12)
            # Radau/LSODA (if LSODA you dont see mdotin and mdotout)
        elif cycle == "4T":
            # LSODA IS THE FASTEST AND PROBABLY MOST ROBUST. RK45 with rtol 1e-12 also works but is a bit slower

            #sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='RK45', y0=x, t_eval=phi,
            #                rtol=5e-11)  # 1e-12 works
            try:
                sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='LSODA', y0=x, t_eval=phi,
                                rtol=1e-7, atol=1e-12)  # 1e-12 needed to not fuck up with latest limits
            except UserWarning as e:
                print(e)
                return np.zeros(nr_output)
        else:
            raise Exception(f'Unknown cycle. The cycle was {cycle}.')

        if i > 0:
            mdiff.append((np.abs(sol.y[4][-1] - m[-1][-1])))
            pdiff.append(np.abs(sol.y[5][-1] - P[-1][-1]))
            Tdiff.append(np.abs(sol.y[0][-1] - T[-1][-1]))
            equdiff.append(np.abs(sol.y[9][-1] - equ[-1][-1]))
            mEP_diff.append(np.abs(sol.y[15][-1] - m_EP[-1][-1]))
            mIP_diff.append(np.abs(sol.y[12][-1] - m_IP[-1][-1]))


        T.append(sol.y[0])
        V.append(sol.y[1])
        Q.append(sol.y[2])
        Q_in.append(sol.y[3])
        m.append(sol.y[4])
        P.append(sol.y[5])
        W.append(sol.y[6])
        m_in.append(sol.y[7])
        mf.append(sol.y[8])
        equ.append(sol.y[9])
        m_out.append(sol.y[10])
        energy_out.append(sol.y[11])
        m_IP.append(sol.y[12])
        T_IP.append(sol.y[13])
        equ_IP.append(sol.y[14])
        m_EP.append(sol.y[15])
        T_EP.append(sol.y[16])
        equ_EP.append(sol.y[17])
        m_in_IP.append(sol.y[18])
        S.append(sol.y[19])
        mdotin.append(sol.y[20])
        mdotout.append(sol.y[21])
        m_out_EP.append(sol.y[22])
        Q_apparent.append((sol.y[23]))
        energy_in.append(sol.y[24])
        fuel_enthalpy_in.append(sol.y[25])

        #if "sweep" not in flags:
        #    print(f"iter {i + 1} of {it}")

        equ_avg = (mf[-1][-1] / m_in_IP[-1][-1]) / far_s

        # this way to calculate far works quite well
        if 'fuel_mass' not in flags:
            if cycle == "2T":
                m_air_inlet_closing = m[-1][np.argwhere(phi >= phi_close_in - cycle_phi)[0][0]] \
                                      / (1 + far_s * equ[-1][np.argwhere(phi >= phi_close_in  - cycle_phi)[0][0]])
                equ_inlet_closing = equ[-1][np.argwhere(phi >= phi_close_in - cycle_phi)[0][0]]
            elif cycle == "4T":
                m_air_inlet_closing = m[-1][-1] \
                                      / (1 + far_s * equ[-1][-1])
                equ_inlet_closing = equ[-1][-1]

            mf_tot_old = mf_tot

            # this is to specify the trapped far
            #mf_tot = (far_goal - equ_inlet_closing * far_s) * m_air_inlet_closing

            # we now care about actual fuel flow and far in exhausts out of the engine
            mf_tot = m_in_IP[-1][-1] * far_goal
            #print(m_in_IP[-1][-1])

            #print(f"Inflow into IP: {m_in_IP[-1][-1]*1000} [g/cycle] ")

            # ensure convergence in added fuel
            mf_diff.append(np.abs(mf_tot - mf_tot_old))
            #Qf is very important. This is the one actually deciding amount of added heat.
            # 99.9% of Qf will be added to the fluid.
            Qf = LHV * mf_tot  # hmmm eta_c or not
        else:
            mf_diff.append(0.0)


        if "fuel_mass" in flags:
            # avg far
            far_avg = mf_tot / m_in_IP[-1][-1]
            def find_tout(t):
                h, _, _, _, _, _, _, _ = thermo.mixture(t[0], p_out, far_avg / far_s, fuel_type)
                return h - energy_out[-1][-1] / m_out_EP[-1][-1]

        else:
            #using far_goal for t_out calculations should be good since that is the average far in exhaust
            def find_tout(t):
                h, _, _, _, _, _, _, _ = thermo.mixture(t[0], p_out, far_goal / far_s, fuel_type)
                return h - energy_out[-1][-1] / m_out_EP[-1][-1]


        try:
            T_out.append(fsolve(find_tout, T[-1][-1])[0])
        except RuntimeWarning as e:
            print(e)
            return np.zeros(nr_output)

        if i > 0:
            T_out_diff.append(np.abs(T_out[-1] - T_out[-2]))

        # Checking if simulation has converged (minimum 5 iterations and lest three diffs should be smaller than limits)
        if i > 4:

            # These values can be motivated at a later stage
            if 'validation' in flags:
                p_lim = 1e-1
                m_lim = 1e-1
                T_lim = 1e-1
                T_out_lim = 1e-1
                equ_lim = 1e-1
                mf_lim = 1e-3
                mEP_lim = 1e-3
                mIP_lim = 1e-3

            else:

                # p_lim normally 1e-1
                p_lim = 1e-3
                m_lim = 1e-7
                T_lim = 1e-3
                T_out_lim = 1e-3
                equ_lim = 1e-5
                mf_lim = 1e-9
                mEP_lim = 1e-9
                mIP_lim = 1e-9

            convergence = True
            for j in range(1,3):
                if pdiff[-j] > p_lim or mdiff[-j] > m_lim or Tdiff[-j] > T_lim \
                        or T_out_diff[-j] > T_out_lim or mf_diff[-j] > mf_lim or equdiff[-j] > equ_lim \
                        or mEP_diff[-j] > mEP_lim or mIP_diff[-j] > mIP_lim:
                    convergence = False

            if convergence:
                end = timer()
                print(f'Simulation has converged after {i + 1} iterations. Runtime of script: {end - start} [s]')
                #print(i, pdiff[-1], mdiff[-1], Tdiff[-1], T_out_diff[-1], mf_diff[-1], equdiff[-1])
                break


        if i + 1 == it:
            if "sweep" in flags:
                print(f'Simulation never converged. p_in, T_in, cr, far_goal, rpm, bore: {p_in * 1e-5, T_in, cr, far_goal, rpm, d}')
                # Return zeros so that the data can be cleaned from simulations that never converged
                return np.zeros(nr_output)
            else:
                end = timer()
                print(f'Simulation never converged. Reached {i + 1} iterations. Runtime of script: {end - start} [s]')
                break


    ## Post processing ##
    # Calculate power and induced mean effective pressure
    n_r = cycle_phi/(2*np.pi)  # crank revolutions per power stroke ( 1 for two-stroke and 2 for four-stroke)
    rps = rpm/60  # revolutions per second
    t_cycle = n_r / rps  # cycle duration [s]
    power = W[-1][-1] / t_cycle  # Indicated power (work per cycle divided by cycle time) [W] (for one cylinder)
    imep = W[-1][-1] / V_d  # indicated mean effective pressure (work per cycle / displacement) [Pa]

    power_engine = W[-1][-1] * cylinders / t_cycle  # Total indicated power for the entire engine
    Vd_tot = V_d * cylinders

    if 'sweep' not in flags:
        fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(d, rpm, s, v_mean, p_in, cr, cylinders, lv_max)
        bmep = imep - fmep

        friction_power = fmep * V_d * rps / n_r  # power lost from piston to crankshaft
        break_power = bmep * V_d * rps / n_r  # total power at the crankshaft

        # Calculate some scavenging things
        purity, residual_fraction, eta_trapping, eta_charging, delivery_ratio, eta_sc = \
            post_processing.scavenging(equ, phi, phi_close_out, phi_open_out, far_s, m_in_IP, rho_in, V_d, m)

    heat_loss_single = Q[-1][-1] / t_cycle

    # Calculate mass flow
    air_flow = m_in_IP[-1][-1] / t_cycle
    fuel_flow = mf_tot / t_cycle
    #fuel_flow = mf[-1][-1] / t_cycle
    out_flow = m_out_EP[-1][-1] / t_cycle
    heat_flow = Q_in[-1][-1] / t_cycle

    eta_th = W[-1][-1] / (mf_tot * LHV)  # work from total fuel heat
    hl = Q[-1][-1] / (mf_tot * LHV)  # heat loss

    # mass of cylinder gasses just before exhaust opening
    m_trapped = m[-1][np.argwhere(phi <= phi_open_out)[-1][0]]

    #eta_th = W[-1][-1] / Q_in[-1][-1]
    #hl = Q[-1][-1] / Q_in[-1][-1]  # heat loss

    # Retrieve maximum pressure and temperature
    p_max = np.max(P[-1])
    T_max = np.max(T[-1])
    equ_trapped = np.max(equ[-1])

    # Maximum pressure before combustion (for calculating injector pressure)
    p_tdc = P[-1][np.argwhere(phi >= 2*np.pi)[0][0]]

    # For multiple cylinders. Values for the entire engine (for example for one V10 engine)
    if "sweep" not in flags:
        friction_power_engine = fmep * Vd_tot * rps / n_r  # power lost from piston to crankshaft
        break_power_engine = break_power * cylinders  # total power at the crankshaft
        friction_losses = fmep_pe_loss * Vd_tot * rps / n_r  # friction losses for total engine all cylinders
        aux_losses = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

        # Calculate entropy for the last cycle
        s_specific = entropy_calc(T[-1], equ[-1], fuel_type, P[-1])

    air_flow_engine = air_flow * cylinders
    fuel_flow_engine = fuel_flow * cylinders

    heat_losses = heat_loss_single * cylinders

    # Fuel air ratio based on the total fuel and air flows
    far_avg = fuel_flow_engine / air_flow_engine

    if "sweep" in flags:
        #removed friction for now when sampling/sweeping
        break_power_engine = 0.0
        friction_losses = 0.0
        aux_losses = 0.0

    print(f"Fuel injected: {mf_tot * 1e6} mg")
    # check for energy conservation
    # get the last values for the arrays for each iteration
    # mass in intake, fuel mass in, mass out outlet
    inflows = np.array([item[-1] for item in m_in_IP])
    fuel_flows = np.array([item[-1] for item in mf])
    outflows = np.array([item[-1] for item in m_out_EP])

    # enthalpy in intake, heat in, enthalpy in fuel, enthalpy out exhaust, heat out walls, work out
    enthalpy_ins = np.array([item[-1] for item in energy_in])
    heat_ins = np.array([item[-1] for item in Q_in])
    fuel_enthalpy_ins = np.array([item[-1] for item in fuel_enthalpy_in])
    enthalpy_outs = np.array([item[-1] for item in energy_out])
    heat_outs = np.array([item[-1] for item in Q])
    works_outs = np.array([item[-1] for item in W])

    term1 = enthalpy_ins[-1] + fuel_enthalpy_ins[-1]
    term2 = enthalpy_outs[-1] + heat_outs[-1] + works_outs[-1]
    diff = np.abs(term1 - term2)

    print(f"Energy conservation: {diff / heat_ins[-1]}")

    if "validation" not in flags:
        # if energy error larger than 0.1% of fuel energy
        if np.abs(diff / heat_ins[-1]) > 0.001:
            print(f"ENERGY NOT CONSERVED!!!!!!: {diff / heat_ins[-1]}")
            return np.zeros(nr_output)


    if "validation" not in flags:

        ## NOX calculations
        # get the heat addition from fuel curve
        dmfdphi = wiebe.dmfdphi_single_mass_vector(phi, m_wiebe, phi_sc, phi_cd, mf_tot)

        """
        num = 10
        factors = np.linspace(0.5, 1.0, num)

        noxss = []
        temps = []
        start = timer()
        for factor in factors:
            # get temperature and mass from reaction zone
            T_z1, m_z1, p_z1, V_z1, lambda_z1, phi_z1, equ_hp, T_z2, m_z2, T_hp = twozone_model.twozone(phi, P[-1], T[-1], V[-1], m[-1], dmfdphi,
                                                                              phi_open_out, phi_sc, LHV, far_s,
                                                                              equ[-1], fuel_type, factor)


            #start = timer()
            no_ppm, dNOdt, no_times = nox_model_cantera.nox_calculations(T_z1, m_z1, p_z1, V_z1, fuel_type, lambda_z1, phi_z1, rpm,
                                              m_out_EP[-1][-1], mf_tot, equ_trapped, m_trapped)
            #end = timer()
            #print(f'NOx calculations done in: {end - start} [s]')
            noxss.append(no_ppm[-1])
            temps.append(no_ppm)

        end = timer()
        print(f'NOx calculations done in: {end - start} [s]')

        fig, ax1 = plt.subplots()
        ax1.plot(factors, noxss)

        fig, ax2 = plt.subplots()
        for i in range(0, num):
            #ax2.plot(temps[i][:], label=f"factor: {i}")
            ax2.plot(temps[i][:])

        #plt.legend()
        plt.show()

        """

        # Greek: 0.84. Heider: 0.9, Scania: 1.0
        factor = 0.84

        # get temperature and mass from reaction zone
        T_z1, m_z1, p_z1, V_z1, lambda_z1, phi_z1, equ_hp, T_z2, m_z2, T_hp, equ_sc = twozone_model.twozone(phi, P[-1], T[-1],
                                                                                                    V[-1], m[-1], dmfdphi,
                                                                                                    phi_open_out, phi_sc,
                                                                                                    LHV, far_s,
                                                                                                    equ[-1], fuel_type,
                                                                                                    factor)

        # start = timer()
        no_ppm, dNOdt, no_times = nox_model_cantera.nox_calculations(T_z1, m_z1, p_z1, V_z1, fuel_type, lambda_z1, phi_z1,
                                                                     rpm,
                                                                     m_out_EP[-1][-1], mf_tot, equ_trapped, m_trapped, equ_sc)


    # post processing
    if "sweep" not in flags:
        if "validation" in flags:
            validation = True
        else:
            validation = False

        if 'save' in flags:
            np.savetxt("simulation_data/P.csv", P[-1], delimiter=",")
            np.savetxt("simulation_data/T.csv", T[-1], delimiter=",")
            np.savetxt("simulation_data/m.csv", m[-1], delimiter=",")
            np.savetxt("simulation_data/equ.csv", equ[-1], delimiter=",")
            np.savetxt("simulation_data/rohr.csv", Q_in[-1], delimiter=",")
            np.savetxt("simulation_data/Qapparent.csv", Q_apparent[-1], delimiter=",")
            np.savetxt("simulation_data/phi.csv", phi, delimiter=",")

        if "plot_validation" in flags:
            from src.misc.plot_output import plot_validation
            plot_validation(phi, P, T, m, equ)

        if "plot_essentials" in flags:
            from piston_engine.src.misc.plot_output import plot_essentials, plot_rohr
            plot_essentials(phi, T, P, m, equ, validation)
            #plot_rohr(phi, Q[-1], Q_in[-1], V[-1], Apiston, dtdphi, d, P[-1], T[-1])


        if "validate_twozone" in flags:
            # validate twozone model with the Heider picture
            from piston_engine.src.misc.plot_output import plot_twozone_validation, plot_no_validation

            # validate twozone model against Heider paper 1998 (from Simulating combustion textbook)
            plot_twozone_validation(phi, T_z1, T_z2, T[-1], P[-1], phi_open_out, phi_sc)
            plot_no_validation(no_ppm, phi_z1)

        elif "validate_nox_diesel_early" in flags:
            # validate NOx model with data from diesel engine (Rakopoulos et al)
            from piston_engine.src.misc.plot_output import plot_nox_diesel_validation
            plot_nox_diesel_validation(phi, T_z1, T_z2, T[-1], P[-1], phi_open_out, phi_sc, mf[-1], no_ppm)

        elif "validate_nox_diesel_late" in flags:
            # validate NOx model with data from diesel engine (Rakopoulos et al)
            from piston_engine.src.misc.plot_output import plot_nox_diesel_validation_late
            plot_nox_diesel_validation_late(phi, T_z1, T_z2, T[-1], P[-1], phi_open_out, phi_sc, mf[-1], no_ppm)


        elif 'validate_scania_highload' in flags:
            # validate NOx model with data from scania engine (KTH msc thesis)
            from piston_engine.src.misc.plot_output import plot_scania_highload
            print(f"Fuel injected: {mf_tot*1e6} mg")
            #print(f"Air sucked in: {m_in_IP[-1][-1] * 1e6} mg")
            #print(f"Displacement: {V_d * 1e6} cm^3")
            plot_scania_highload(phi, P[-1], dmfdphi, LHV, Q_apparent[-1])

        elif 'validate_scania_lowload' in flags:
            # validate NOx model with data from scania engine (KTH msc thesis)
            from piston_engine.src.misc.plot_output import plot_scania_lowload
            print(f"Fuel injected: {mf_tot*1e6} mg")
            #print(f"Air sucked in: {m_in_IP[-1][-1] * 1e6} mg")
            #print(f"Displacement: {V_d * 1e6} cm^3")
            plot_scania_lowload(phi, P[-1], dmfdphi, LHV, Q_apparent[-1])

        else:
            # no validation
            if "plot_twozone" in flags:
                from piston_engine.src.misc.plot_output import plot_twozone_full, plot_twozone_only
                plot_twozone_full(phi, T_z1, T_z2, T[-1], phi_open_out, phi_sc)
                plot_twozone_only(phi_z1, T_z1, T_z2, T_hp, m_z1, m_z2)



        if "plot_pv" in flags:
            from src.misc.plot_output import plot_pvts
            # plot_pv(P[-1], V[-1])
            # plot_ts(T[-1], S[-1])
            plot_pvts(P[-1], V[-1], T[-1], s_specific, S[-1])

        if "plot_convergence" in flags:
            from piston_engine.src.misc.plot_output import (plot_convergence, plot_massconservation,
                                                            plot_energyconservation, plot_convergence2)

            plot_convergence(pdiff, Tdiff, mdiff, equdiff, T_out_diff, mf_diff)
            plot_convergence2(mEP_diff, mIP_diff)
            plot_massconservation(inflows, fuel_flows, outflows)
            plot_energyconservation(enthalpy_ins, heat_ins, fuel_enthalpy_ins, enthalpy_outs, heat_outs, works_outs)

        if "plot_all" in flags:
            from src.misc.plot_output import plot_energy, plot_convergence, plot_diagrams, plot_progress
            from src.misc.plot_output import plot_essentials, plot_manifolds, plot_details, plot_massflows
            plot_essentials(phi, T, P, m, equ, validation)
            plot_massflows(phi, m_in, mf, mdotin, mdotout, V, validation)
            plot_energy(phi, W, Q, m_in, Q_in, mf)
            plot_convergence(pdiff, Tdiff, mdiff, equdiff)
            plot_manifolds(phi, equ_IP, m_IP, T_IP, equ_EP, m_EP, T_EP)

        if 'output_all' and 'validation' in flags:
            from src.misc.output import output_thermo_validation, output_power_validation, \
                output_scavenging_validation, output_efficiencies_validation
            output_thermo_validation(phi, P, T, T_out[-1], air_flow, m_in_IP, fuel_flow, mf)
            output_power_validation(power, imep, friction_power, fmep, break_power, bmep, heat_loss_single)
            output_efficiencies_validation(eta_th, hl)
            output_scavenging_validation(purity, residual_fraction, eta_trapping,
                                         eta_charging, delivery_ratio, eta_sc, equ_avg)

        if 'output_power' and 'validation' in flags:
            from piston_engine.src.misc.output import output_power_validation
            output_power_validation(power_engine, imep, friction_power_engine, fmep, break_power_engine, bmep,
                                    heat_loss_single)

        if 'output_all' in flags and not validation:
            from piston_engine.src.misc.output import output_thermo, output_power, output_efficiencies
            output_thermo(phi, P, T, T_out[-1], air_flow, m_in_IP, fuel_flow, mf)
            output_power(power, imep, friction_power, fmep, break_power, bmep, heat_loss_single, equ_avg)
            output_efficiencies(eta_th, hl)

        if 'output_power' in flags and not validation:
            from piston_engine.src.misc.output import output_power
            output_power(power_engine, imep, friction_power_engine, fmep, break_power_engine, bmep, heat_loss_single,
                         equ_avg)




    return T_out[-1], break_power_engine, eta_th, air_flow_engine, p_max, T_max, far_avg, equ_trapped,\
        power_engine, friction_losses, aux_losses, heat_losses, p_tdc, out_flow, no_ppm[-1], imep
