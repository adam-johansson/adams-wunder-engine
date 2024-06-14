import pandas as pd


X = pd.read_csv('../piston_engine/sampled_data/h2/X.csv', index_col=0)
y = pd.read_csv('../piston_engine/sampled_data/h2/y.csv', index_col=0)

mask = y.eff > 0
y_cleaned = y[mask]
X_cleaned = X[mask]


pd.DataFrame.to_csv(X_cleaned, '../piston_engine/sampled_data/h2/X_cleaned.csv')
pd.DataFrame.to_csv(y_cleaned, '../piston_engine/sampled_data/h2/y_cleaned.csv')




