from CCE.src.components import nozzle
from thermo import fuel_props

far_s, LHV = fuel_props("H2")
"""
p1 = 48323.15
t1 = 266.725
pa = 26200.5
equ = 0.0
m = 207.948
cfg = 0.995
cd = 0.99
fuel_type = "H2"


p1 = 37863.19
t1 = 638.492
pa = 26200.5
equ = 0.0066 / far_s
m = 5.951
cfg = 0.977
cd = 0.9564
fuel_type = "H2"
"""
p1 = 35961.95
t1 = 529.97
pa = 26200.5
equ = 0.0066 / far_s
m = 5.951
cfg = 0.977
cd = 0.9564
fuel_type = "H2"


F, v2_id, v2, error = nozzle(p1, t1, pa, equ, m, cfg, cd, fuel_type)


print(F, v2, v2_id)