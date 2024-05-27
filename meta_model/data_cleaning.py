from numpy import genfromtxt
import pandas as pd

headers = ['p_in', 'T_in', 'cr', 'd', 'far', 'ratio', 'v_mean']
xt = pd.read_csv('../piston_engine/surrogate_data/xt.csv', header=None, names=headers)
xval = pd.read_csv('../piston_engine/surrogate_data/xval.csv', header=None, names=headers)
headers = ['T_out', 'eta', 'air_flow', 'p_max', 'T_max', 'power_indicated', 'heat_loss', 'p_tdc']
yt = pd.read_csv('../piston_engine/surrogate_data/yt.csv', header=None, names=headers)
yval = pd.read_csv('../piston_engine/surrogate_data/yval.csv', header=None, names=headers)

mask = yt.eta > 0
yt_cleaned = yt[mask]
xt_cleaned = xt[mask]
mask = yval.eta > 0
yval_cleaned = yval[mask]
xval_cleaned = xval[mask]

pd.DataFrame.to_csv(xt_cleaned, '../piston_engine/surrogate_data/xt_cleaned.csv')
pd.DataFrame.to_csv(yt_cleaned, '../piston_engine/surrogate_data/yt_cleaned.csv')
pd.DataFrame.to_csv(xval_cleaned, '../piston_engine/surrogate_data/xval_cleaned.csv')
pd.DataFrame.to_csv(yval_cleaned, '../piston_engine/surrogate_data/yval_cleaned.csv')



