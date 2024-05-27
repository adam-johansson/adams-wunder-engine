import pandas as pd


# import the splitted csv files
xt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xt_cleaned.csv', index_col=0)
yt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yt_cleaned.csv', index_col=0)
xval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xval_cleaned.csv', index_col=0)
yval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yval_cleaned.csv', index_col=0)


# concatenating along rows
x = pd.concat([xval, xt], axis=0, ignore_index=True)
y = pd.concat([yval, yt], axis=0, ignore_index=True)


pd.DataFrame.to_csv(x, '../piston_engine/surrogate_data/backup/h2_validated_woschni_18/x_cleaned.csv')
pd.DataFrame.to_csv(y, '../piston_engine/surrogate_data/backup/h2_validated_woschni_18/y_cleaned.csv')