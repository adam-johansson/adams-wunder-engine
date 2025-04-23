import numpy as np

cycle = "4T"

fuel = "H2"

thermo = "NASA"

# cooling = "Hohenberg"
cooling = "Woschni"
# cooling = "H2"

opposed = False
cr = 10.306640294734263

cylinders = 12  # V12

# piston
d = 0.134  # diameter / bore
# s = 0.074676  # stroke
v_mean = 10.0  # mean velocity
# l_con = 0.182067  # rod length
bsr = 1.0  # bore stroke ratio
lms = 1 / (
    2 * 1.7
)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 1126246  # inlet pressure
T_in = 793.5  # inlet temperature
p_ratio = 1.45  # pressure ratio after and before engine

# Heat transfer
Twall = 500  # Wall temperature
Tpiston = 600
Thead = 600
Twalls = [Twall, Tpiston, Thead]


ch = 1.8  # multiplier to decrease/increase heat transfer

# Inlet valve
# phi_open_in = (725/180)*np.pi  # from Argonne paper
# phi_close_in = (935/180)*np.pi  # from Argonne paper

# outlet valve
# phi_open_out = (495/180)*np.pi  # from Argonne paper
# phi_close_out = (750/180)*np.pi  # from Argonne paper

phi_open_in = (715 / 180) * np.pi  # pressure rise
phi_close_in = (918 / 180) * np.pi  # pressure rise

# outlet valve
phi_open_out = (515 / 180) * np.pi  # for pressure rise
phi_close_out = (729 / 180) * np.pi  # for pressure rise


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d
cd = 0.8

eta_c = 1.0

far_goal = 0.01148


wiebe_type = "Single"
# wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 1.75  # from validation italian

phi_sc = (361 / 180) * np.pi  # angle at combustion start  from validation italian
phi_cd = (
    35 / 180
) * np.pi  # angle related to combustion duration from validation italian

T_fuel = 360
p_fuel = 300e5

it = 400

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function
