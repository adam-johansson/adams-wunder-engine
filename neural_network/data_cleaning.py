import pandas as pd


X = pd.read_csv('./input_data/H2/x.csv', index_col=0)
y = pd.read_csv('./input_data/H2/y.csv', index_col=0)

mask = y.eff > 0
y_cleaned = y[mask]
X_cleaned = X[mask]


pd.DataFrame.to_csv(X_cleaned, './input_data/H2/x_cleaned.csv')
pd.DataFrame.to_csv(y_cleaned, './input_data/H2/y_cleaned.csv')




