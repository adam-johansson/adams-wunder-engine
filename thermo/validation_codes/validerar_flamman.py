import matplotlib.pyplot as plt
import numpy as np

from thermo import flame_temp_inhouse, flame_temp_cea, flame_temp_cantera


fuel_type = "H2"

t_soc = 700
Psc = 1e5

equ_sc = 0.0
equ_combustion = 1.0

adam = []
cea = []
cantera = []

for equ_combustion in np.linspace(0.1, 1.0):

    t_adam = flame_temp_inhouse(t_soc, equ_sc, equ_combustion, fuel_type)

    t_cea = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc, equ_combustion)

    t_cantera_jetA = flame_temp_cantera(t_soc, Psc, equ_sc, equ_combustion, fuel_type)

    adam.append(t_adam)
    cea.append(t_cea)
    cantera.append(t_cantera_jetA)


plt.plot(adam, label="adam")
plt.plot(cea, label="cea")
plt.plot(cantera, label="cantera")
plt.legend()
plt.show()


# fuel_type = "CH4"
# t_cantera_CH4 = flame_temp_cantera(t_soc, Psc, equ_sc, equ_combustion, fuel_type)
