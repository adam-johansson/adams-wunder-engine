

from thermo import flame_temp_inhouse, flame_temp_cea


fuel_type = "H2"

t_soc = 298
Psc = 5e5

equ_sc = 0.0

t_adam = flame_temp_inhouse(t_soc, equ_sc, fuel_type)

t_cea = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc)


print(t_adam)

print(t_cea)