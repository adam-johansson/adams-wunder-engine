import numpy as np

# THIS IS THE PAPER TYSK2 NOT ARGONNE
# H2 DIRECT INJECTION A HIGHLY PROMISING COMBUSTION CONCEPT

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

#cooling = "Hohenberg"
cooling = "Woschni"
#cooling = "H2"

opposed = False
cr = 10.5  # geometric compression ratio (Kaiser used 9.6 ToC and 6.4 TO and 18 in Cruise)

cylinders = 1  # V12

# piston
d = 0.086  # diameter / bore
# s = 0.074676  # stroke
v_mean = 5.7333  # mean velocity 2000 rpm
# l_con = 0.182067  # rod length
bsr = 0.086 / 0.086  # bore stroke ratio
# rpm = 4000  # revolutions per minute
lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 1e5  # inlet pressure
T_in = 300  # inlet temperature
p_ratio = 0.9  # pressure ratio after and before engine

# Heat transfer
Twall = 500          # Wall temperature
Tpiston = 500
Thead = 500
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (725/180)*np.pi  # working
phi_close_in = (935/180)*np.pi  # working


# outlet valve
phi_open_out = (495/180)*np.pi  # working
phi_close_out = (750/180)*np.pi  # working


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.0095
cd = 0.8

eta_c = 0.999

throttle = 0.0298 / 2.7


#wiebe_type = "Single"
wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 2

phi_sc = (362/180)*np.pi  # angle at combustion start  THIS WORKED WITH SINGLE #345
phi_cd = (20/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #55

T_fuel = 300
p_fuel = 200e5

it = 40

mf_tot = 1.5e-4