import pandas as pd
import numpy as np

database = pd.read_csv("desafio001.csv")
print(f"Antes:\n", database.head())

database = database.drop_duplicates()
database = database.drop("endereço", axis=1) #O '1' são as colunas
print(f"Depois:\n", database.head())
