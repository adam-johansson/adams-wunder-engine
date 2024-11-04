import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


X = pd.read_csv('./input_data/H2_adaptive_large/x.csv', index_col=0)
y = pd.read_csv('./input_data/H2_adaptive_large/y.csv', index_col=0)


print(np.shape(X))

# remove all the data points with zeros
mask = y.p_max > 0
print(f"Number of data points that was removed during sampling {np.count_nonzero(mask == 0)}")
y_cleaned = y[mask]
X_cleaned = X[mask]

# insight into the data
max_pressure = np.max(y_cleaned.p_max)
min_pressure = np.min(y_cleaned.p_max)

max_power = np.max(y_cleaned.power)
min_power = np.min(y_cleaned.power)

max_heat = np.max(y_cleaned.heat_loss)
min_heat = np.min(y_cleaned.heat_loss)

max_flow = np.max(y_cleaned.air_flow)
min_flow = np.min(y_cleaned.air_flow)

max_eff = np.max(y_cleaned.eff)
min_eff = np.min(y_cleaned.eff)

max_t = np.max(y_cleaned.T_max)
min_t = np.min(y_cleaned.T_max)

max_tout = np.max(y_cleaned.T_out)
min_tout = np.min(y_cleaned.T_out)

max_ptdc = np.max(y_cleaned.p_tdc)
min_ptdc = np.min(y_cleaned.p_tdc)

print(f"Maximum peak pressure is {max_pressure} bar")
print(f"Minimum peak pressure is {min_pressure} bar")

print(f"Maximum power is {max_power} kW")
print(f"Minimum power is {min_power} kW")

print(f"Maximum heat loss is {max_heat} kW")
print(f"Minimum heat loss is {min_heat} kW")

print(f"Maximum air flow is {max_flow} kg/s")
print(f"Minimum air flow is {min_flow} kg/s")

print(f"Maximum efficiency is {max_eff * 100} %")
print(f"Minimum efficiency is {min_eff * 100} %")

print(f"Maximum peak temperature is {max_t} K")
print(f"Minimum peak temperature is {min_t} K")

print(f"Maximum outlet temperature is {max_tout} K")
print(f"Minimum outlet temperature is {min_tout} K")

print(f"Maximum pressure at top dead centre is {max_ptdc} bar")
print(f"Minimum pressure at top dead centre is {min_ptdc} bar")

# count how many data points have peak pressure below 300 bar
print(f"Number of data points with peak pressure over 300 bar {np.count_nonzero(y.p_max > 300)}")


plt.plot(y_cleaned.p_max, 'o', markersize=1)
plt.ylabel("peak pressure [bar]")
plt.xlabel("data point")
plt.show()


pd.DataFrame.to_csv(X_cleaned, './input_data/H2_adaptive_large/x_cleaned.csv')
pd.DataFrame.to_csv(y_cleaned, './input_data/H2_adaptive_large/y_cleaned.csv')




