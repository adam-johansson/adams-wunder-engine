import numpy as np

# THIS IS THE ONE FOR ICAS. 2500 rpm, 08 equivalence ratio
# THIS IS THE ONE FOR ICAS

cycle = "4T"

fuel = "H2"

thermo = "NASA"

cooling = "Woschni"
# cooling = "H2"
# cooling = "Hohenberg"

opposed = False
cr = 10  # geometric compression ratio

cylinders = 1

# piston
d = 0.087  # diameter / bore (specified)
# s = 0.074676  # stroke
# rpm = v_mean / (2 * s) * 60
v_mean = 2501 * 2 * 0.085 / 60  # mean velocity 2500 rpm
# l_con = 0.182067  # rod length
bsr = 0.087 / 0.085  # bore stroke ratio ( specified)

lms = 1 / (
    2 * 1.7
)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor) (1.7 original)

# inlet and outlet conditions
p_in = 1.01325e5  # inlet pressure  (1.27 + 495K in)
T_in = 360.0  # inlet temperature 450 + working angles
p_ratio = (
    1.01325e5 / p_in
)  # pressure ratio after and before engine (assumption based on 1% loss in inlet and outlet)

# Heat transfer
Twall = 450  # Wall temperature
Tpiston = 450
Thead = 450
Twalls = [Twall, Tpiston, Thead]


ch = 1.8  # multiplier to increase heat transfer for hydrogen operation

# Inlet valve

# 1.7 lms + 465 open_out + 1/2.649 = 5.46 IMEP

# camshaft has half speed of engine (110 LSA means 220 angle between exhaust opening and intake opening)

phi_open_in = ((725) / 180) * np.pi  # working 725
phi_close_in = ((905) / 180) * np.pi  # working 935

# outlet valve
phi_open_out = (496 / 180) * np.pi  # working 496
phi_close_out = ((750) / 180) * np.pi  # working 750

# phi_open_in = ((678)/180)*np.pi  # working 725
# phi_close_in = ((946)/180)*np.pi  # working 935

# outlet valve
# phi_open_out = (458/180)*np.pi  # working 496
# phi_close_out = ((716)/180)*np.pi  # working 750


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 1
valve_type = "valve"

lv_max = 0.005
cd = 0.8

# eta_c = 0.999
eta_c = 1.0

# 2.3 matcher total heat för 11 cd och m = 1.1
throttle = 0.02918 * 0.8

phi_sc = (361.0 / 180) * np.pi  # angle at combustion start 361
phi_cd = (35 / 180) * np.pi  # angle related to combustion duration

T_fuel = 300
p_fuel = 150e5

wiebe_type = "Single"

# This is for Kaisers wiebe function
wa = 11.0
wm = 0.6

# this is for single wiebe function
m_wiebe = 1.75  # 1.75

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function

it = 40

mf_tot = 1.5e-4

premixed = False
