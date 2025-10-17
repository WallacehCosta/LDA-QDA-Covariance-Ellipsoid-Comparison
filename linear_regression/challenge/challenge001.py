import numpy as np
import pandas as pd

dataSet = pd.read_csv('students.csv', delimiter=';')
x = dataSet.values
num_cols = dataSet.select_dtypes(include=['number']).columns

from sklearn.impute import SimpleImputer
num_cat = SimpleImputer(missing_values=np.nan, strategy='median')
dataSet[num_cols] = num_cat.fit_transform(dataSet[num_cols])

x = dataSet.values

print(x)
