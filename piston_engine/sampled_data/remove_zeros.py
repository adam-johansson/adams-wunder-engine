

import pandas as pd

data = pd.read_csv("./jetA/data.csv", index_col=0)

data_cleaned = data[data.eff != 0]

# Writing data to file
data_cleaned.to_csv("./jetA/data_cleaned.csv")
