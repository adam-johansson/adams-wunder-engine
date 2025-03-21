import numpy as np

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

#cooling = "Hohenberg"
cooling = "Woschni"
#cooling = "H2"

# premixed or direct_injection or external?
premixed = False

opposed = False
cr = 6

cylinders = 1  # for sampling data use 1

# piston
d = 0.10  # diameter / bore
# s = 0.074676  # stroke
v_mean = 15.0  # mean velocity
# l_con = 0.182067  # rod length
bsr = 1.0  # bore stroke ratio
lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 10e5  # inlet pressure
T_in = 750  # inlet temperature
#T_in = 1000
p_ratio = 0.9  # pressure ratio after and before engine

# Heat transfer
Twall = 500          # Wall temperature
Tpiston = 600
Thead = 600
Twalls = [Twall, Tpiston, Thead]


ch = 1.4  # multiplier to decrease/increase heat transfer (THIS WAS 1.8 before)

# Inlet valve
phi_open_in = (715/180)*np.pi  # pressure rise
phi_close_in = (918/180)*np.pi  # pressure rise

# outlet valve
phi_open_out = (515/180)*np.pi  # for pressure rise
phi_close_out = (729/180)*np.pi  # for pressure rise


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d
cd = 0.8

eta_c = 0.999

far_goal = 0.0291755 / 1.3


wiebe_type = "Single"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 1.75  # from validation italian

phi_sc = (361/180)*np.pi  # angle at combustion start  from validation italian
phi_cd = (35/180)*np.pi  # angle related to combustion duration from validation italian

T_fuel = 450
p_fuel = 300e5

it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function