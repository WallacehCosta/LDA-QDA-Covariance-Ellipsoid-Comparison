import numpy as np
import pandas as pd

database = pd.read_csv("desafio004.csv")
print(database.head())

#database.loc[condição, onde aplicar a condição] = o que eu quero aplicar/mudar
database.loc[database["salário"] < 2000, "salário"] = "SM"

print(database.head())
