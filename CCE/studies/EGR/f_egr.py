import numpy as np
import matplotlib.pyplot as plt


# fuel-air-ratio without egr
#far_goal = np.linspace(0.02,0.06)
far_goal = 0.03

# egr rate of exhaust gas into the piston engine
egr_rate = np.linspace(0.0,0.5)


#far_needed_egr1 = (1-egr_rate-far_goal)/(1+egr_rate) - np.sqrt(0.25*((far_goal + egr_rate - 1)/(1+egr_rate))**2 - far_goal/(1+egr_rate) )

#far_needed_egr2 = (1-egr_rate-far_goal)/(1+egr_rate) + np.sqrt(0.25*((far_goal + egr_rate - 1)/(1+egr_rate))**2 - far_goal/(1+egr_rate) )

far_needed_egr1 = -(1 - egr_rate - far_goal)/(2*(1-egr_rate)) + np.sqrt(( (1 - egr_rate - far_goal)**2 / (4 * (1-egr_rate)**2     )   )   + far_goal/(1-egr_rate)   )

plt.plot(egr_rate, far_needed_egr1)
plt.show()