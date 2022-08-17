import pandas as pd
from sklearn.metrics import mean_squared_error


# baseline benchmark (naive forecast)

data = pd.read_csv("daily_min_temp.csv")

# prediction is the next value
print(mean_squared_error(data.iloc[:,1][:-2], data.iloc[:, 1].shift(-1)[:-2]))
