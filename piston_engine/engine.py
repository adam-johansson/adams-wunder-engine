import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import time
import sys

from piston_engine.src.piston import valve_isentrop, walls, wiebe, port_isentrop, twozone_model, nox_model, nox_model_alternative, nox_model_hcci

from piston_engine.src.misc import post_processing
from piston_engine.src.misc.entropy import entropy_calc
from numba import njit

import thermo

from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp

dPdphi_temp = 0


def run_piston_engine(input, flags):
    p_in = input['p_in']
    T_in = input['T_in']
    equ_ratio_in = input['equ_in']
    p_ratio = input['p_ratio']
    cycle = input['cycle']
    cooling = input['cooling']
    opposed = input['opposed']
    cr = input['cr']
    d = input['bore']
    bsr = input['bsr']
    v_mean = input['v_mean']
    lms = input['lms']
    Twalls = input['Twalls']
    ch = input['ch']
    valve_timings = input['valve_timings']
    n_valve = input['n_valve']
    lv_max = input['lv_max']
    cd = input['cd']
    eta_c = input['eta_c']
    mf_tot = input['mf_tot']
    m_wiebe = input['m_wiebe']
    phi_sc = input['phi_sc']
    phi_cd = input['phi_cd']
    T_fuel = input['T_fuel']
    p_fuel = input['p_fuel']
    it = input['it']
    wiebe_type = input['wiebe_type']
    valve_type = input['valve_type']
    far_goal = input['far_goal']
    cylinders = input['cylinders']
    fuel_type = input['fuel']
    c1 = input['c1']
    c4 = input['c4']
    c5 = input['c5']

    mode = input["mode"]

    if mode == "DI":
        premixed = False
        HCCI = False
    elif mode == "SI":
        premixed = True
        HCCI = False
    elif mode == "HCCI":
        premixed = True
        HCCI = True
    else:
        print(f"Unknown mode of operation: {mode}")

        
    # outlet pressure
    p_out = p_ratio * p_in

    Twalls = np.array(Twalls)

    phi_open_in = valve_timings[0]
    phi_close_in = valve_timings[1]
    phi_open_out = valve_timings[2]
    phi_close_out = valve_timings[3]

    # number of outputs from the function
    nr_output = 20

    s = d/bsr  # stroke retrieved from bore stroke ratio
    l_con = s/(2*lms)  # connecting rod length
    rpm = 60*v_mean/(2*s)  # rpm from mean velocity

    V_d = d ** 2 * (np.pi / 4) * s  # displacement
    #print(f"Displacement: {V_d*1000} liter")

    if opposed:
        # Opposed piston has double volume
        V_d = V_d * 2

    V_clearance = V_d / (cr - 1)
    V_max = V_clearance + V_d

    far_s, LHV = thermo.fuel_props(fuel_type)

    # assuming pure air in intake for direct injection and a fuel-air mixture for external mixture formation

    # Now we can specifiy the equivalence ratio of the intake air
    h_in, _, _, _, R_in, _, _, _ = thermo.mixture(t=T_in, p=p_in, equivalence_ratio=equ_ratio_in, fuel_type=fuel_type, include_fuel_in_reactants=premixed, fuel_air_equ_ratio=far_goal/far_s)

    rho_in = p_in / (R_in * T_in)

    phi_ec = (phi_sc + phi_cd)  # angle at combustion end

    if fuel_type == 'jetA':
        cp_fuel, h_fuel, _, _ = thermo.polynomials.JETA_L(T_fuel)
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
        # I DONT THINK THIS WORKS
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
            # mass of charge when using inlet density and displacement
            m_air_theo = V1[0] * rho_in

        if premixed:
            # Get fuel mass fraction from thermodynamic properties
            _, _, _, _, _, mass_fraction_fuel = thermo.mass_fractions(
                0.0, fuel_type, premixed, far_goal / far_s
            )

            # Calculate fuel mass needed for next cycle based on trapped mixture
            mf_tot = m0 * mass_fraction_fuel
        else:
            # this should account for diluted intake air
            mf_tot = m_air_theo * (far_goal - equ_ratio_in * far_s) / (1 + equ_ratio_in * far_s)
            #mf_tot = far_goal * m_air_theo


        # far_tot is used for init of exhaust port equivalance ratio
        far_tot = far_goal

        # Qf is used in some cases but for single_mass Wiebe the mf_tot is used for the fuel addition 
        Qf = LHV * mf_tot  #hmmm eta_c or not
        #print(mf_tot)

        #print(f"Equ in intake: {far_goal / far_s}")


    def dxdphi(phi, x, Pref, Tref, Vref, Pmotor, Vmotor):

        # solve a system of ODEs for pressure, temperature, volume
        # assign ode to vector element

        #print("Var scaled", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15],
        #      x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25])

        T = x[0] * inv_scaler[0]  # temperature
        V = x[1] * inv_scaler[1] # volume
        #Q = x[2]  # heat loss
        #Qin = x[3]  # heat in
        m = x[4] * inv_scaler[4]  # mass
        P = x[5] * inv_scaler[5]
        # work
        #m_in = x[7]
        #mf = x[8]
        equ = x[9] * inv_scaler[9]
        #m_out = x[10]
        #energy_out = x[11]
        m_IP = x[12] * inv_scaler[12]
        T_IP = x[13] * inv_scaler[13]
        equ_IP = x[14] * inv_scaler[14]
        m_EP = x[15] * inv_scaler[15]
        T_EP = x[16] * inv_scaler[16]
        equ_EP = x[17] * inv_scaler[17]
        # m_in_IP = x[18]
        # entropy_cyl = x[19]
        # mdotin = 20
        # mdotout = 21
        #mout_EP = x[22]
        #H_inIP = x[23]
        #H_fuel_in = x[24]

        if premixed and equ > far_goal / far_s:
            equ = far_goal / far_s

        if premixed and equ_IP > far_goal / far_s:
            equ_IP = far_goal / far_s

        if premixed and equ_EP > far_goal / far_s:
            equ_EP = far_goal / far_s

        # Gas properties inside the cylinder
        h, u, cp, cv, R, gamma, entropy, _ = thermo.mixture(T, P, equ, fuel_type,
                                                            include_fuel_in_reactants=premixed,
                                                            fuel_air_equ_ratio=far_goal/far_s)

        # Intake port values
        h_IP, u_IP, cp_IP, cv_IP, R_IP, gamma_IP, entropy_IP, _ = thermo.mixture(T_IP, p_in, equ_IP, fuel_type,
                                                                                 include_fuel_in_reactants=premixed,
                                                                                 fuel_air_equ_ratio=far_goal/far_s)

        # Exhaust port values
        h_EP, u_EP, cp_EP, cv_EP, R_EP, gamma_EP, entropy_EP, _ = thermo.mixture(T_EP, p_out, equ_EP, fuel_type,
                                                                                 include_fuel_in_reactants=premixed,
                                                                                 fuel_air_equ_ratio=far_goal/far_s)

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

        Awalls = np.array([Awall, Apiston, Ahead])

        if cooling == "Woschni":
            if np.mod(phi, cycle_phi) > phi_sc and np.mod(phi, cycle_phi) < phi_open_out:
                firing = True
            else:
                firing = False

            if cycle == "4T":
                if phi_close_in - cycle_phi < np.mod(phi, cycle_phi) < phi_open_out:
                    hp_phase = True
                else:
                    hp_phase = False
            elif cycle == "2T":
                if phi_close_out < phi < phi_open_out:
                    hp_phase = True
                else:
                    hp_phase = False
            else:
                raise Exception(f'Unknown cycle. The cycle was {cycle}.')

            if fuel_type == "jetA":
                dqdphi, alpha = walls.dqdphi_woschni(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                             hp_phase, firing)
            else:
                # for hydrogen
                dqdphi, alpha = walls.dqdphi_woschni_h2(T, P, V, gamma, Twalls, Awalls, v_mean, d, V_d, ref, dtdphi,
                                             hp_phase, firing)



        elif cooling == "Hohenberg":
            dqdphi = walls.dqdphi_hohenberg(T, P, V, Twalls, Awalls, v_mean, dtdphi)
        elif cooling == "Michl":
            dqdphi = walls.dqdphi_michl(d, T, P, R, V, Twalls, Awalls, v_mean, dtdphi)
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

        # convert to phi from time
        dmindphi = dmindphi * dtdphi
        dmoutdphi = dmoutdphi * dtdphi

        if premixed:
            # no fuel is injected (dmfdphi is only used for the other calculations)
            dmdphi = dmindphi - dmoutdphi
        else:
            dmdphi = dmindphi - dmoutdphi + dmfdphi

        #x[20] = dmindphi
        #x[21] = dmoutdphi

        # INTAKE PORT CALCULATIONS    
        if dmindphi > 0:
            h_out_IP = h_IP
            equ_out_IP = equ_IP
        else:
            h_out_IP = h
            equ_out_IP = equ

        h_in_IP = h_in
        equ_in_IP = equ_ratio_in

        dellRdellequ_IP, delludellequ_IP = \
            thermo.equivalence_derivative(equ_IP, T_IP, p_in, fuel_type, premixed, far_goal/far_s)

        #term1 = T_IP + h_out_IP / cv_IP + \
        #    (1 + equ_IP * far_s) * (equ_IP - equ_out_IP) / (cv_IP * (1 + equ_out_IP * far_s)) * delludellequ_IP - \
        #    (T_IP / R_IP) * (1 + equ_IP * far_s) * (equ_IP - equ_out_IP) / (1 + equ_out_IP * far_s)

        #term2 = T_IP + h_in_IP / cv_IP + \
        #    (equ_IP * (1 + equ_IP * far_s) / cv_IP) * delludellequ_IP - \
        #    (T_IP * equ_IP / R_IP) * (1 + equ_IP * far_s) * dellRdellequ_IP

        # NOTE: if simulations have become slower, make these equations simplified (h - u e.g.) look at the terms above
        term1 = h_out_IP - u_IP + m_IP * cv_IP * R_IP * T_IP - \
            (1 + equ_IP * far_s) * (delludellequ_IP - cv_IP * m_IP * T_IP * dellRdellequ_IP) * ((equ_out_IP - equ_IP)/(1 + equ_out_IP * far_s)) 
        
        term2 = h_in_IP - u_IP + m_IP * cv_IP * R_IP * T_IP - \
            (1 + equ_IP * far_s) * (delludellequ_IP - cv_IP * m_IP * T_IP * dellRdellequ_IP) * ((equ_in_IP - equ_IP)/(1 + equ_in_IP * far_s)) 


        # mass inflow to intake port
        dmindphi_IP = dmindphi * term1 / term2

        # Continuity equation of the intake port
        dmdphi_IP = dmindphi_IP - dmindphi

        # Change of equivalence ratio in the intake port
        #dequdphi_IP = ((1 + equ_IP * far_s) / m_IP) * (
        #        (equ_IP - equ_out_IP) * dmindphi / (1 + equ_out_IP * far_s) - equ_IP * dmindphi_IP)


                        # Change in equivalence ratio in cylinder
        dequdphi_IP = ((1 + equ_IP * far_s) / m_IP) * (((equ_in_IP - equ_IP) / (1 + equ_in_IP * far_s)) * dmindphi_IP
                                                    - ((equ_out_IP - equ_IP) / (1 + equ_out_IP * far_s)) * dmindphi
                    )
        


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
            thermo.equivalence_derivative(equ_EP, T_EP, p_out, fuel_type, premixed, far_goal/far_s)

        # Change of equivalence ratio in the exhaust port
        #dequdphi_EP = dmoutdphi * (1 + equ_EP * far_s) * (equ_in_EP - equ_EP) / (1 + equ_in_EP * far_s)
        dequdphi_EP = dmoutdphi * (1 + equ_EP * far_s) * (equ_in_EP - equ_EP) / (m_EP * (1 + equ_in_EP * far_s))

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

        dellRdellequ, delludellequ = thermo.equivalence_derivative(equ, T, P, fuel_type, premixed, far_goal/far_s)

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
        if premixed:
            dudphi = (- dqdphi - P * dVdphi - u * dmdphi +
                      dmindphi * h_in_cyl - dmoutdphi * h_out_cyl) / m
        else:
            dudphi = (- dqdphi - P * dVdphi - u * dmdphi + dmfdphi * h_fuel +
                      dmindphi * h_in_cyl - dmoutdphi * h_out_cyl) / m

        dTdphi = (dudphi - delludellequ * dequdphi) / cv

        # Ideal gas law
        dPdphi = (m * R * dTdphi + R * T * dmdphi + m * T * dRdphi - P * dVdphi) / V

        # if P > 150e5:
        #    dPdphi = dPdphi * 1.05

        # Entropy (Gibbs equation)
        dsdphi = dudphi / T + dVdphi * ( P / T)

        derivatives = np.array([dTdphi, dVdphi, dqdphi, dqfdphi, dmdphi, dPdphi,
                P * dVdphi , dmindphi, dmfdphi, dequdphi, dmoutdphi, h_out_EP * dmoutdphi_EP,
                dmdphi_IP, dTdphi_IP, dequdphi_IP, dmdphi_EP, dTdphi_EP, dequdphi_EP,
                dmindphi_IP, dsdphi, 0.0, 0.0, dmoutdphi_EP, h_in_IP * dmindphi_IP, h_fuel * dmfdphi])


        # scale the derivatives
        derivatives_scaled = derivatives * scaler

        return derivatives_scaled

        #return derivatives


    # from initial guess of fuel air ratio
    equ_EP0 = far_tot / far_s
    equ0 = equ_IP0 = equ_ratio_in

    # Init simulation
    x = np.array([T0, V1[0], 0.0, 0.0, m0, P0, 0.0, 0.0, 0.0, equ0, 0.0, 0.0, m0, T_in, equ_IP0, m0, T_in, equ_EP0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial state value (0.1 m0 in IP and 1e-5 m0 in EP)

    scale_ODE = True
    if scale_ODE == True:

        # Scaling factors for the variables to keep them of the same order of magnitude
        scaler = np.array([1e-3, 1e3, 1e-3, 1e-3, 1e2, 1e-5, 1e-3, 1e2, 1e3, 1e1, 1e2, 1e-3, 1e2,
                           1e-2, 1e1, 1e2, 1e-2, 1e1, 1e2, 1e-3, 1e-2, 1e-2, 1e2, 1e-3, 1e-3])

        # Scaling factors for the variables to keep them of the same order of magnitude
        inv_scaler = 1 / scaler

    else:
        scaler = np.ones((25))
        inv_scaler = scaler

    # Scaled variables init
    x_scaled = x * scaler  # initial state value

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
    T_EP_diff = []
    T_IP_diff = []

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
                 m_EP[-1][-1], T_EP[-1][-1], equ_EP[-1][-1], 0, S[-1][-1], 0.0, 0.0, 0.0, 0.0, 0.0]

            x_scaled = x * scaler
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
            print(i)

            sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='LSODA', y0=x_scaled, t_eval=phi,
                            rtol=1e-9, atol=1e-9)  # standard is rtol 1e-7 and atol 1e-10 (no scaling)

            # Radau/LSODA (if LSODA you dont see mdotin and mdotout)
        elif cycle == "4T":
            # LSODA IS THE FASTEST AND PROBABLY MOST ROBUST. RK45 with rtol 1e-12 also works but is a bit slower

            #sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='RK45', y0=x, t_eval=phi,
            #                rtol=5e-11)  # 1e-12 works
            try:
                sol = solve_ivp(dxdphi, args=woschni_args, t_span=(min(phi), max(phi)), method='LSODA', y0=x_scaled, t_eval=phi,
                                rtol=1e-8, atol=1e-8)  # 1e-8 and 1e-10 are standard
            except UserWarning as e:
                print(e)
                return np.zeros(nr_output)
        else:
            raise Exception(f'Unknown cycle. The cycle was {cycle}.')

        # the variables but scaled
        scaled_results = sol.y

        # real values
        results = scaled_results * inv_scaler[:, np.newaxis]

        if i > 0:
            #mdiff.append((results[4][-1] - m[-1][-1]))
            #diff.append(results[5][-1] - P[-1][-1])
            #diff.append(results[0][-1] - T[-1][-1])
            #equdiff.append(results[9][-1] - equ[-1][-1])
            #mEP_diff.append(results[15][-1] - m_EP[-1][-1])
            #mIP_diff.append(results[12][-1] - m_IP[-1][-1])
            #T_EP_diff.append(results[16][-1] - T_EP[-1][-1])
            #T_IP_diff.append(results[13][-1] - T_IP[-1][-1])

            # relative limits
            mdiff.append((results[4][-1] - m[-1][-1]) / results[4][-1]  )
            pdiff.append((results[5][-1] - P[-1][-1]) / results[5][-1])
            Tdiff.append((results[0][-1] - T[-1][-1]) / results[0][-1])
            equdiff.append((results[9][-1] - equ[-1][-1]) / results[9][-1])
            mEP_diff.append((results[15][-1] - m_EP[-1][-1]) / results[15][-1])
            mIP_diff.append((results[12][-1] - m_IP[-1][-1]) / results[12][-1])
            T_EP_diff.append((results[16][-1] - T_EP[-1][-1]) / results[16][-1])
            T_IP_diff.append((results[13][-1] - T_IP[-1][-1]) / results[13][-1])

        T.append(results[0])
        V.append(results[1])
        Q.append(results[2])
        Q_in.append(results[3])
        m.append(results[4])
        P.append(results[5])
        W.append(results[6])
        m_in.append(results[7])
        mf.append(results[8])
        equ.append(results[9])
        m_out.append(results[10])
        energy_out.append(results[11])
        m_IP.append(results[12])
        T_IP.append(results[13])
        equ_IP.append(results[14])
        m_EP.append(results[15])
        T_EP.append(results[16])
        equ_EP.append(results[17])
        m_in_IP.append(results[18])
        S.append(results[19])
        mdotin.append(results[20])
        mdotout.append(results[21])
        m_out_EP.append(results[22])
        energy_in.append(results[23])
        fuel_enthalpy_in.append(results[24])

        # inflow of pure air
        if premixed:
            m_in_air = (m_in_IP[-1][-1] / (1 + far_goal))
        else:
            m_in_air = m_in_IP[-1][-1]
        equ_avg = (mf[-1][-1] / m_in_air) / far_s

        # Simple way to calculate fuel-air ratio - works quite well
        # This is to calculate how much fuel is to be injected each cycle
        
        if 'fuel_mass' not in flags:

            mf_tot_old = mf_tot

            if premixed:

                if cycle == "2T":
                    # Find inlet closing conditions - extract the complex indexing once
                    closing_idx = np.argwhere(phi >= phi_close_in - cycle_phi)[0][0]

                    total_mass_at_closing = m[-1][closing_idx]
                    equ_inlet_closing = equ[-1][closing_idx]

                elif cycle == "4T":
                    equ_inlet_closing = equ[-1][-1]
                    total_mass_at_closing = m[-1][-1]


                # Get fuel mass fraction from thermodynamic properties
                _, _, _, _, _, mass_fraction_fuel = thermo.mass_fractions(
                    equ_inlet_closing, fuel_type, premixed, far_goal / far_s
                )

                # Calculate fuel mass needed for next cycle based on trapped mixture
                mf_tot = total_mass_at_closing * mass_fraction_fuel

            else:
                # Non-premixed: fuel flow based on actual exhaust fuel-air ratio
                # fuel air ratio is based on the intake pure air flow

                mf_tot = m_in_IP[-1][-1] * (far_goal - equ_ratio_in * far_s) / (1 + equ_ratio_in * far_s)

            # Track convergence of fuel mass iteration
            mf_diff.append(np.abs(mf_tot - mf_tot_old))

            # Calculate total heat addition
            Qf = LHV * mf_tot


        else:
            # Fuel mass externally specified - no iteration needed
            mf_diff.append(0.0)

        if "fuel_mass" in flags:
            # avg far
            far_avg = mf_tot / m_in_IP[-1][-1]
            def find_tout(t):
                h, _, _, _, _, _, _, _ = thermo.mixture(t, p_out, far_avg / far_s, fuel_type,
                                                        include_fuel_in_reactants=premixed, fuel_air_equ_ratio=far_goal/far_s)
                return h - energy_out[-1][-1] / m_out_EP[-1][-1]

        else:
            #using far_goal for t_out calculations should be good since that is the average far in exhaust
            def find_tout(t):
                h, _, _, _, _, _, _, _ = thermo.mixture(t, p_out, far_goal / far_s, fuel_type, include_fuel_in_reactants=premixed, fuel_air_equ_ratio=far_goal/far_s)
                return h - energy_out[-1][-1] / m_out_EP[-1][-1]


        try:
            T_out.append(brentq(find_tout, 200, 6000))
        except RuntimeWarning as e:
            print(e)
            return np.zeros(nr_output)

        if i > 0:
            T_out_diff.append(np.abs(T_out[-1] - T_out[-2]))

        # Checking if simulation has converged (minimum 5 iterations and lest three diffs should be smaller than limits)
        if i > 4:
            # print(i, pdiff[-1], mdiff[-1], Tdiff[-1], T_out_diff[-1], mf_diff[-1], equdiff[-1], mIP_diff[-1], mEP_diff[-1])

            # These values can be motivated at a later stage
            if 'validation' in flags:
                p_lim = 1e-2
                m_lim = 1e-2
                T_lim = 1e-2
                T_out_lim = 1e-2
                equ_lim = 1e-2
                mf_lim = 1e-4
                mEP_lim = 1e-4
                mIP_lim = 1e-4

            else:

                # p_lim normally 1e-1
                #these values are for the absolute limits
                #p_lim = 1e-2
                #m_lim = 1e-7
                #T_lim = 1e-3
                #T_out_lim = 1e-3
                #equ_lim = 1e-5
                #mf_lim = 1e-8
                #mEP_lim = 1e-6
                #mIP_lim = 1e-6

                # these values are for relative limits
                p_lim = 1e-3
                m_lim = 1e-3
                T_lim = 1e-3
                T_out_lim = 1e-3
                equ_lim = 1e-3
                mf_lim = 1e-3
                mEP_lim = 1e-3
                mIP_lim = 1e-3

            convergence = True
            non_converged_params = []

            # Check each parameter and track which ones don't converge
            for j in range(1, 3):
                if np.abs(pdiff[-j]) > p_lim:
                    convergence = False
                    if f'pdiff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'pdiff[-{j}]', np.abs(pdiff[-j]), p_lim))

                if np.abs(mdiff[-j]) > m_lim:
                    convergence = False
                    if f'mdiff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'mdiff[-{j}]', np.abs(mdiff[-j]), m_lim))

                if np.abs(Tdiff[-j]) > T_lim:
                    convergence = False
                    if f'Tdiff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'Tdiff[-{j}]', np.abs(Tdiff[-j]), T_lim))

                if np.abs(T_out_diff[-j]) > T_out_lim:
                    convergence = False
                    if f'T_out_diff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'T_out_diff[-{j}]', np.abs(T_out_diff[-j]), T_out_lim))

                if np.abs(mf_diff[-j]) > mf_lim:
                    convergence = False
                    if f'mf_diff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'mf_diff[-{j}]', np.abs(mf_diff[-j]), mf_lim))

                if np.abs(equdiff[-j]) > equ_lim:
                    convergence = False
                    if f'equdiff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'equdiff[-{j}]', np.abs(equdiff[-j]), equ_lim))

                if np.abs(mEP_diff[-j]) > mEP_lim:
                    convergence = False
                    if f'mEP_diff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'mEP_diff[-{j}]', np.abs(mEP_diff[-j]), mEP_lim))

                if np.abs(mIP_diff[-j]) > mIP_lim:
                    convergence = False
                    if f'mIP_diff[-{j}]' not in [param[0] for param in non_converged_params]:
                        non_converged_params.append((f'mIP_diff[-{j}]', np.abs(mIP_diff[-j]), mIP_lim))

            if convergence:
                end = timer()
                # print(f'Simulation has converged after {i + 1} iterations. Runtime of script: {end - start} [s]')
                # print(i, pdiff[-1], mdiff[-1], Tdiff[-1], T_out_diff[-1], mf_diff[-1], equdiff[-1])
                break


        if i + 1 == it:
            if "sweep" in flags:
                print(
                    f'Simulation never converged. p_in, T_in, cr, far_goal, rpm, bore, p_ratio: {p_in * 1e-5, T_in, cr, far_goal, rpm, d, p_ratio}')
                # Show final non-converged parameters for debugging
                if non_converged_params:
                    print('Final non-converged parameters:')
                    for param_name, current_val, limit in non_converged_params:
                        print(
                            f'  {param_name}: {current_val:.2e} (limit: {limit:.2e}, ratio: {current_val / limit:.2f})')
                # Return zeros so that the data can be cleaned from simulations that never converged
                return np.zeros(nr_output)
            else:
                end = timer()
                print(f'Simulation never converged. Reached {i + 1} iterations. Runtime of script: {end - start} [s]')
                # Show final non-converged parameters for debugging
                if non_converged_params:
                    print('Final non-converged parameters:')
                    for param_name, current_val, limit in non_converged_params:
                        print(
                            f'  {param_name}: {current_val:.2e} (limit: {limit:.2e}, ratio: {current_val / limit:.2f})')
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
        #fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(d, rpm, s, v_mean, p_in, cr, cylinders, lv_max)
        #bmep = imep - fmep

        #friction_power = fmep * V_d * rps / n_r  # power lost from piston to crankshaft
        #break_power = bmep * V_d * rps / n_r  # total power at the crankshaft

        tot_loss_power, aux_loss_power, friction_loss_power = post_processing.friction_patton(d, rpm, s, v_mean, p_in, cr, cylinders,
                                                                        lv_max, cycle)

        fmep = tot_loss_power * n_r / (V_d * rps)

        break_power_engine = power_engine - tot_loss_power

        bmep = break_power_engine * n_r / (V_d * rps)

        if not premixed:
            # Calculate some scavenging things
            purity, residual_fraction, eta_trapping, eta_charging, delivery_ratio, eta_sc = \
                post_processing.scavenging(equ, phi, phi_close_out, phi_open_out, far_s, m_in_IP, rho_in, V_d, m)

    heat_loss_single = Q[-1][-1] / t_cycle

    # only account for the swept volume of the piston
    V_swept = V_d - V_clearance

    # Calculate mass flow
    if premixed:
        # inflow of pure air
        air_flow = (m_in_IP[-1][-1] / (1 + far_goal)) / t_cycle

        # calculating properties for the pure air
        _, _, _, _, R_in_air, _, _, _ = thermo.mixture(t=T_in, p=p_in)

        # density of the air at the inlet (before mixing with the fuel)
        rho_in_air = p_in / (R_in_air * T_in)

        # pure air sucked in vs theoretical max
        volume_eff = (m_in_IP[-1][-1] / (1 + far_goal)) / (V_swept * rho_in_air)

    else:
        # this is the mass flow of gas into the piston engine. it could either be pure air or a diluted mixture from EGR
        massflow_intake = m_in_IP[-1][-1] / t_cycle
        air_flow = massflow_intake / (1 + equ_ratio_in * far_s)
        # air sucked in vs theoretical max
        volume_eff = m_in_IP[-1][-1] / (rho_in * V_swept)

  
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

    if "sweep" not in flags:
        #friction_power_engine = fmep * Vd_tot * rps / n_r  # power lost from piston to crankshaft
        #break_power_engine = break_power * cylinders  # total power at the crankshaft
        #friction_losses = fmep_pe_loss * Vd_tot * rps / n_r  # friction losses for total engine all cylinders
        #aux_losses = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

        # Calculate entropy for the last cycle
        s_specific = entropy_calc(T[-1], equ[-1], fuel_type, P[-1])

    air_flow_engine = air_flow * cylinders
    fuel_flow_engine = fuel_flow * cylinders
    massflow_intake_engine = massflow_intake * cylinders

    heat_losses = heat_loss_single * cylinders

    # Fuel air ratio based on the total fuel and air flows
    far_avg = fuel_flow_engine / air_flow_engine

    if "sweep" in flags:
        #removed friction for now when sampling/sweeping
        break_power_engine = 0.0
        friction_loss_power = 0.0
        aux_loss_power = 0.0

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

    if premixed:
        # In premixed operation, no fuel is injected and therefore
        # no enthalpy from the fuel (it is in the mixture already)
        term1 = enthalpy_ins[-1]
    else:
        term1 = enthalpy_ins[-1] + fuel_enthalpy_ins[-1]
    term2 = enthalpy_outs[-1] + heat_outs[-1] + works_outs[-1]
    diff = np.abs(term1 - term2)

    #print(f"Energy conservation: {diff / heat_ins[-1] * 100} %")

    # if energy error larger than 0.1% of fuel energy
    if "sweep" in flags:
        if np.abs(diff / heat_ins[-1]) > 0.001:
            print(f"ENERGY NOT CONSERVED!!!!!!: {diff / heat_ins[-1]}")
            return np.zeros(nr_output)

    if (cycle == "4T" and far_goal > 0.0):
        ## NOX calculations
        # get the heat addition from fuel curve
        dmfdphi = wiebe.dmfdphi_single_mass_vector(phi, m_wiebe, phi_sc, phi_cd, mf_tot)


        # factor 0.885 and lambda = 1.1 BÄST
        # factor 0.865 and lambda = 1.09 BRA EXAKT SAMMA SOM 1.08
        # factor 0.865 and lambda = 1.08 BRA TYP LIKA BRA SOM 1.1 
        # factor 0.8575 and lambda = 1.07 BRA
        # factor 0.8425 and lambda = 1.06 Ganska BRA
        # factor 0.8425 and lambda = 1.05 RÄTT BRA
        # factor 0.8425 and lambda = 1.04 HELT OK
        # factor 0.84 and lambda = 1.03 SÅDÄR
        # factor 0.83 and lambda = 1.02 GÅR EJ
        # factor 0.845 and lambda = 1.0 
        factor = 0.845

        start = timer()
        # get temperature and mass from reaction zone (zone 1 is hot zone)
        # if premixed, lambda_z1 = lambda_gl
        T_z1, m_z1, p_z1, V_z1, lambda_z1, phi_z1, equ_hp, T_z2, m_z2, T_hp, equ_sc, T_flame, T_sc, p_sc = twozone_model.twozone(phi, P[-1], T[-1],
                                                                                                    V[-1], m[-1], dmfdphi,
                                                                                                    phi_open_out, phi_sc,
                                                                                                    LHV, far_s,
                                                                                                    equ[-1], fuel_type,
                                                                                                    factor, premixed)
        end = timer()
        #print(f'Runtime of twozone calculations: {end - start} [s]')
        T_max_twozone = np.max(T_z1)
        start = timer()

        ## FOR HCCI we use single zone for NOX
        if HCCI:
                # use single zone values for the nox calculations
                hp_mask = (phi > phi_sc) & (phi < phi_open_out)
                phi_z1 = phi[hp_mask]
                p_z1 = P[-1][hp_mask]
                T_z1 = T[-1][hp_mask]
                V_z1 = V[-1][hp_mask]
                m_z1 = m[-1][hp_mask]
                equ_z1 = equ[-1][hp_mask]


        
        if HCCI:
            no_ppm, dNOdt, no_times, EI_nox, m_NO = nox_model_hcci.nox_calculations(T_z1, p_z1, V_z1, equ_z1, fuel_type, phi_z1,
                                                rpm,
                                                m_out_EP[-1][-1], mf_tot, equ_trapped, m_trapped, equ_sc)
        else:
            # when sampling data use nox_model
            # it is more stable for weird input parameters
            # otherwise use nox_model_alternative.nox_calculations 
            no_ppm, dNOdt, no_times, EI_nox, m_NO = nox_model.nox_calculations(T_z1, p_z1, V_z1, fuel_type, lambda_z1, phi_z1,
                                                                        rpm,
                                                                        m_out_EP[-1][-1], mf_tot, equ_trapped, m_trapped, equ_sc)

        end = timer()

        # THIS IS ONLY WHEN WE WANT TO COMPARE NOX FROM KINETICS TO EQUILIBIRUM
        nox_equilibrium = False
        if nox_equilibrium:
            from piston_engine.src.piston import nox_model_equi
            from piston_engine.src.misc import plot_output, save_nox_equ
            if HCCI:
                no_ppm_equi = nox_model_equi.nox_hcci(T_z1, p_z1, m_z1, equ_z1, equ_trapped, m_out_EP[-1][-1])
            else:
                no_ppm_equi = nox_model_equi.nox(T_z1, p_z1, m_z1, lambda_z1, equ_trapped, m_out_EP[-1][-1])

            plot_output.plot_no_with_equi(phi,phi_open_out,phi_sc,no_ppm, no_ppm_equi)
            plt.show()


            save_nox_equ.save_nox(phi, phi_open_out,phi_sc,no_ppm, no_ppm_equi, p_z1, T_z1)



        #print(f'Runtime of NOx calculations: {end - start} [s]')



        # calculate specific NOX emissions (g of NO per kWh of work produced)

        # convert from J to kwH and from kg to g
        nox_spec = (m_NO[-1] / W[-1][-1]) * (3600 * 1e3) * 1e3


    elif cycle == "2T":
        EI_nox = 999
        no_ppm = np.array([999])
        nox_spec = 999
        T_max_twozone = 0
        T_flame = 0
        T_sc = 0
        p_sc = 0


    # post processing
    if "sweep" not in flags:
        if "validation" in flags:
            validation = True
        else:
            validation = False

        if 'save' in flags:
            np.savetxt("piston_engine/simulation_data/P.csv", P[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/T.csv", T[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/m.csv", m[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/V.csv", V[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/equ.csv", equ[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/Q_in.csv", Q_in[-1], delimiter=",")
            np.savetxt("piston_engine/simulation_data/phi.csv", phi, delimiter=",")

        if "plot_validation" in flags:
            from src.misc.plot_output import plot_validation
            plot_validation(phi, P, T, m, equ)

        if "plot_essentials" in flags:
            from piston_engine.src.misc.plot_output import plot_essentials, plot_rohr
            plot_essentials(phi, T, P, m, equ, validation)
            #plot_rohr(phi, Q[-1], Q_in[-1], V[-1], Apiston, dtdphi, d, P[-1], T[-1])


        if "validate_twozone" in flags:
            # validate twozone model with the Heider picture
            from piston_engine.src.misc.plot_output import plot_twozone_validation, plot_no_validation, plot_twozone_validation_final

            # validate twozone model against Heider paper 1998 (from Simulating combustion textbook)
            #plot_twozone_validation(phi, T_z1, T_z2, T[-1], P[-1], phi_open_out, phi_sc)
            #plot_no_validation(no_ppm, phi_z1)

            plot_twozone_validation_final(phi, T_z1, T_z2, T[-1], P[-1], phi_open_out, phi_sc, no_ppm, phi_z1)

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

        elif 'fit_water_paper' in flags:
            # validate NOx model with data from scania engine (KTH msc thesis)
            from piston_engine.src.misc.plot_output import val_water_paper_h2

            val_water_paper_h2(phi, P[-1], dmfdphi, LHV, Q_apparent[-1])

        elif 'fit_newcastle' in flags:
            # validate NOx model with data from scania engine (KTH msc thesis)
            from piston_engine.src.misc.plot_output import val_newcastle

        elif 'validate_chalmers' in flags:
            from piston_engine.src.misc.plot_output import val_chalmers

            val_chalmers(phi, P[-1], T[-1], Q_in[-1], V[-1], equ[-1], premixed)

        else:
            # no validation
            if "plot_twozone" in flags:
                from piston_engine.src.misc.plot_output import plot_twozone_full, plot_twozone_only, plot_no, plot_addedfuel
                plot_no(phi, phi_open_out, phi_sc, no_ppm)
                plot_addedfuel(phi, dmfdphi)
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
            plot_convergence2(mEP_diff, mIP_diff, T_EP_diff, T_IP_diff)
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
            from piston_engine.src.misc.output import output_thermo_validation, output_power_validation, \
                output_scavenging_validation, output_efficiencies_validation
            output_thermo_validation(phi, P, T, T_out[-1], air_flow, m_in_IP, fuel_flow, mf)
            output_power_validation(power, imep, friction_loss_power, fmep, break_power_engine, bmep, heat_loss_single)
            output_efficiencies_validation(eta_th, hl)
            output_scavenging_validation(purity, residual_fraction, eta_trapping,
                                         eta_charging, delivery_ratio, eta_sc, equ_avg)

        if 'output_power' and 'validation' in flags:
            from piston_engine.src.misc.output import output_power_validation
            output_power_validation(power_engine, imep, friction_loss_power, fmep, break_power_engine, bmep,
                                    heat_loss_single)

        if 'output_all' in flags and not validation:
            from piston_engine.src.misc.output import output_thermo, output_power, output_efficiencies
            output_thermo(phi, P, T, T_out[-1], air_flow, m_in_IP, fuel_flow, mf)
            output_power(power, imep, friction_loss_power, fmep, break_power_engine, bmep, heat_loss_single, equ_avg)
            output_efficiencies(eta_th, hl)

        if 'output_power' in flags and not validation:
            from piston_engine.src.misc.output import output_power
            output_power(power_engine, imep, friction_power_engine, fmep, break_power_engine, bmep, heat_loss_single,
                         equ_avg)
            

    output_dict = {
        "T_out": T_out[-1],
        "break power": break_power_engine,
        "eta_th": eta_th,
        "intake massflow": massflow_intake_engine, 
        "intake airflow": air_flow_engine,
        "fuel flow": fuel_flow_engine,
        "peak pressure": p_max,
        "peak temperature": T_max,
        "far": far_avg,
        "equ_trapped": equ_trapped,
        "indicated power": power_engine,
        "friction_loss_power": friction_loss_power,
        "aux_loss_power": aux_loss_power,
        "heat_loss": heat_losses,
        "p_tdc": p_tdc,
        "out_flow": out_flow,
        "no_ppm": no_ppm[-1],
        "imep": imep,
        "EI_NO": EI_nox,
        "volume_eff": volume_eff,
        "nox_spec": nox_spec,
        "peak temperature hot zone": T_max_twozone,
        "flame temperature": T_flame,
        "T start of combustion": T_sc,
        "p start of combustion": p_sc,
        "pressure trace": P[-1],
        "temperature trace": T[-1],
        "crank angle trace": phi,
        "mass trace": m[-1],
        "volume trace": V[-1],
    }


    return output_dict