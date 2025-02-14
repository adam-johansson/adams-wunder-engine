

from thermo import flame_temp_inhouse, flame_temp_cea


fuel_type = "jetA"

t_soc = 298
Psc = 5e5

equ_sc = 0.0
equ_combustion = 1.0

t_adam = flame_temp_inhouse(t_soc, equ_sc, equ_combustion, fuel_type)

t_cea = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc, equ_combustion)


print(t_adam)

print(t_cea)