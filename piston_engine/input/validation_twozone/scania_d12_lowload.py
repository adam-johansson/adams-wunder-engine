import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = 'jetA'

cooling = "Hohenberg"
#cooling = "Woschni"

opposed = False

# given in the paper
cr = 17.0

cylinders = 1  # V12

# piston
# given
d = 0.127  # diameter

# given
v_mean = (1200 / 60 ) * 2 * d  # rpm rpm = v_mean / (2 * s) * 60 this is 1400 rpm

# given
stroke = 0.154
bsr = d / stroke  # bore stroke ratio

rod_stroke_ratio = 0.255 / stroke
lms = 1 / (2 * rod_stroke_ratio)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
#p_in = 1.93e5  # inlet pressure
p_in = 1.01325e5 + 0.754e5  # inlet pressure
T_in = 345  # inlet temperature
p_ratio = 0.9  # pressure ratio after and before engine

# Heat transfer
Twall = 400          # Liner temperature
Tpiston = 400
Thead = 400
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# inlet valve
phi_open_in = (718.0/180)*np.pi  # given
phi_close_in = (931.0/180)*np.pi  # given

# outlet valve
phi_open_out = (506.0/180)*np.pi  # given
phi_close_out = (734.0/180)*np.pi  # given

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

# deciding how much fuel to injected (we specify mf_tot)
mf_tot = 70 * 1e-6
far_goal = 0.008


# coefficeint of flow in valves
cd = 0.8

# this is for single wiebe function
m_wiebe = 0.1
phi_sc = (358.0/180)*np.pi  # angle at combustion start
phi_cd = (40/180)*np.pi  # angle related to combustion duration 43


T_fuel = 298
p_fuel = 2500e5


# valves per side
n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d


### OBSOLOTE PARAMETERS (OR RARELY USED) ###

# 99.9 should be used
eta_c = 0.999

wiebe_type = "Single_mass"

# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar

it = 100

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function