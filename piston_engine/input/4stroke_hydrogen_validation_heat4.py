import numpy as np

# LOSS ANALYSIS OF A DIRECT INJECTION HYDROGEN COMBUSTION ENGINE
# THIS IS THE ONE FOR ICAS

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

cooling = "Woschni"
#cooling = "H2"
#cooling = "Hohenberg"

opposed = False
cr = 12  # geometric compression ratio

cylinders = 1

# piston
d = 0.110  # diameter / bore (specified)
# s = 0.074676  # stroke
v_mean = 6.7995  # mean velocity 1000 rpm (specified in tysk2)  4.533 = 1000
# l_con = 0.182067  # rod length
bsr = 0.110 / 0.136  # bore stroke ratio ( specified)

lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor) (1.7 original)

# inlet and outlet conditions
p_in = 3.4e5  # inlet pressure
T_in = 390  # inlet temperature ASSUMPTION 20 deg Celsius 390
p_ratio = 1  # pressure ratio after and before engine (assumption based on 1% loss in inlet and outlet)

# Heat transfer
Twall = 500          # Wall temperature
Tpiston = 500
Thead = 500
Twalls = [Twall, Tpiston, Thead]


ch = 1.4  # multiplier to increase heat transfer for hydrogen operation

# Inlet valve

# 1.7 lms + 465 open_out + 1/2.649 = 5.46 IMEP

# camshaft has half speed of engine (110 LSA means 220 angle between exhaust opening and intake opening)

phi_open_in = ((725)/180)*np.pi  # working 725
phi_close_in = ((935)/180)*np.pi  # working 935

# outlet valve
phi_open_out = (500/180)*np.pi  # working 496
phi_close_out = ((730)/180)*np.pi  # working 750

#phi_open_in = ((678)/180)*np.pi  # working 725
#phi_close_in = ((946)/180)*np.pi  # working 935

# outlet valve
#phi_open_out = (458/180)*np.pi  # working 496
#phi_close_out = ((716)/180)*np.pi  # working 750


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.01
cd = 0.8

#eta_c = 0.999
eta_c = 1.0

# 2.3 matcher total heat för 11 cd och m = 1.1
throttle = 0.0298 / 1.9

phi_sc = (358/180)*np.pi  # angle at combustion start 358
phi_cd = (32/180)*np.pi  # angle related to combustion duration  32

T_fuel = 300
p_fuel = 150e5

wiebe_type = "Single"

# This is for Kaisers wiebe function
wa = 11.0
wm = 0.6

# this is for single wiebe function
m_wiebe = 1.5  #0.6

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function

it = 40

mf_tot = 1.5e-4