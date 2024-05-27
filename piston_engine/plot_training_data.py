import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


xt = np.genfromtxt('surrogate_data/xt.csv', delimiter=',')
xval = np.genfromtxt('surrogate_data/xval.csv', delimiter=',')
yt = np.genfromtxt('surrogate_data/yt.csv', delimiter=',')
yval = np.genfromtxt('surrogate_data/yval.csv', delimiter=',')


t_data = np.concatenate((xt, yt), axis=1)

headers = np.array(['p_in', 'T_in', 'cr', 'bore', 'far', 'p_ratio', 'T_out', 'eta_th', 'mdot', 'p_max', 'T_max',
                    'P_ind', 'P_heat', 'p_tdc'])

t_data = pd.DataFrame(t_data, columns=headers)

print(t_data)

plt.plot(t_data['p_in'], t_data['P_ind'])
plt.show()





