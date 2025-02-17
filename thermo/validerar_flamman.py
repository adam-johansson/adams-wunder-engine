

from thermo import flame_temp_inhouse, flame_temp_cea, flame_temp_cantera


fuel_type = "jetA"

t_soc = 298
Psc = 10e5

equ_sc = 0.3
equ_combustion = 1.0

t_adam = flame_temp_inhouse(t_soc, equ_sc, equ_combustion, fuel_type)

t_cea = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc, equ_combustion)

t_cantera = flame_temp_cantera(t_soc, Psc, equ_sc, equ_combustion)


print(t_adam)

print(t_cea)

print(t_cantera)