from statistics import median

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

database = pd.read_csv('desafio007.csv')
print("Antes:\n", database)

#column = database.iloc[1:, -1].values
#database.fillna(median(column), implace=True)

#filter = database.select_dtypes(include=np.number).columns
database.fillna(median(database.iloc[1:, -1].values), inplace = True)

print("\nDepois:\n", database)
