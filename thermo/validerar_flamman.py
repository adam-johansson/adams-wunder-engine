

from thermo import flame_temp_inhouse, flame_temp_cea, flame_temp_cantera


fuel_type = "jetA"

t_soc = 1000
Psc = 1e5

equ_sc = 0.2
equ_combustion = 0.8

t_adam = flame_temp_inhouse(t_soc, equ_sc, equ_combustion, fuel_type)

t_cea = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc, equ_combustion)


t_cantera_jetA = flame_temp_cantera(t_soc, Psc, equ_sc, equ_combustion, fuel_type)

fuel_type = "CH4"
t_cantera_CH4 = flame_temp_cantera(t_soc, Psc, equ_sc, equ_combustion, fuel_type)

print(t_adam)

print(t_cea)

print(t_cantera_jetA)

print(t_cantera_CH4)

