import pandas as pd
import numpy as np


X = pd.read_csv('./input_data/H2_narrow/x.csv', index_col=0)
y = pd.read_csv('./input_data/H2_narrow/y.csv', index_col=0)

data = pd.concat([X, y], axis = 1)
pd.DataFrame.to_csv(data, './input_data/H2_narrow/data.csv')


print(np.shape(X))

# remove negative efficiencies
mask = y.eff > 0
print(f"Number of data points with negative efficiency {np.count_nonzero(mask == 0)}")
y_cleaned = y[mask]
X_cleaned = X[mask]

# remove pressures over 400 bar
mask = y_cleaned.p_max < 400
print(f"Number of data points with peak pressure under 300 bar: {np.count_nonzero(mask)}")
print(f"Number of data points with peak pressure over 300 bar: {np.count_nonzero(mask == 0)}")
y_cleaned = y_cleaned[mask]
X_cleaned = X_cleaned[mask]

# remove inlet temp under 500 K
mask = X_cleaned.T_in > 500
print(f"Number of data points with inlet temp over 500 K: {np.count_nonzero(mask)}")
print(f"Number of data points with inlet temp under 500 K: {np.count_nonzero(mask == 0)}")
y_cleaned = y_cleaned[mask]
X_cleaned = X_cleaned[mask]


# remove pressure rise over 1.3
mask = X_cleaned.PI < 1.3
print(f"Number of data points with pressure rise under 1.3: {np.count_nonzero(mask)}")
print(f"Number of data points with pressure rise over 1.3: {np.count_nonzero(mask == 0)}")
y_cleaned = y_cleaned[mask]
X_cleaned = X_cleaned[mask]

# remove bore smaller than 100mm
mask = X_cleaned.bore > 0.1
print(f"Number of data points with bore larger than 100mm: {np.count_nonzero(mask)}")
print(f"Number of data points with bore smaller than 100mm: {np.count_nonzero(mask == 0)}")
y_cleaned = y_cleaned[mask]
X_cleaned = X_cleaned[mask]


# remove far under 3 air fuel equic
mask = X_cleaned.far > 0.015
print(f"Number of data points with far larger than 3 : {np.count_nonzero(mask)}")
print(f"Number of data points with far smaller than 3: {np.count_nonzero(mask == 0)}")
y_cleaned = y_cleaned[mask]
X_cleaned = X_cleaned[mask]


data_cleaned = pd.concat([X_cleaned, y_cleaned], axis = 1)



pd.DataFrame.to_csv(X_cleaned, './input_data/H2_verynarrow/x_cleaned.csv')
pd.DataFrame.to_csv(y_cleaned, './input_data/H2_verynarrow/y_cleaned.csv')




