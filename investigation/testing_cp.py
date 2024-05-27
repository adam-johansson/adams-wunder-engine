import numpy as np
from CoolProp.CoolProp import PropsSI

# Coolant (FC)
#f2.fluid = 'INCOMP::ZM-50%'
#f2.dataSource = 'Coolprop_incomp'

p1 = 1e5
ts = np.linspace(15, 25, 100)
for t1 in ts:


    rho = PropsSI('D', 'T', t1, 'P', p1, 'Hydrogen')


    print(rho, t1)
