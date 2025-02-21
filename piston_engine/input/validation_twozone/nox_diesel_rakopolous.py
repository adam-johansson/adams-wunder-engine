import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = 'jetA'

cooling = "Hohenberg"
#cooling = "Woschni"

opposed = False

# given in the paper
cr = 19.81

cylinders = 1

# piston
# given
d = 0.08026  # diameter

# given
rpm = 2500
v_mean = (rpm / 60 ) * 2 * d

# given
bsr = 0.08026 / 0.08890  # bore stroke ratio

lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions (1.5 bar)
p_in = 1.0125e5 * 0.85  # inlet pressure
T_in = 298  # inlet temperature
p_ratio = 1.0  # pressure ratio after and before engine

ch = 1.0  # multiplier to decrease or increase heat transfer

# Heat transfer
# given
Twall = 450          # Liner temperature
Tpiston = 450
Thead = 450
Twalls = [Twall, Tpiston, Thead]

# inlet valve (given)
phi_open_in = (712.0/180)*np.pi  # 712
phi_close_in = (942.0/180)*np.pi  # 905

# outlet valve
phi_open_out = (480/180)*np.pi  # 510
phi_close_out = (732.0/180)*np.pi  # 725


far_goal = 0.041

# this is for single wiebe function
# parameters gotten from minimising MSE
m_wiebe = 2.85

phi_sc = (340.48/180)*np.pi  # angle at combustion start
phi_cd = (44.6/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #43
T_fuel = 298
p_fuel = 2500e5




### RARELY CHANGED PARAMETERS ###

wiebe_type = "Single_mass"

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

# valves per side
n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d

# coefficeint of flow in valves
cd = 0.8

# 99.9 should be used
eta_c = 0.999



### PARAMETERS BELOW ARE OBSOLETE ###
# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar
it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function