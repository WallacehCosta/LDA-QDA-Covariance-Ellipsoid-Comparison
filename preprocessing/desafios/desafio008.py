import numpy as np
import pandas as pd
import datetime as dt

dataset = pd.read_csv("desafio008.csv")
print("Antes:\n", dataset)

dataset["data_venda"] = pd.to_datetime(dataset["data_venda"])
filter = dataset[dataset["data_venda"].dt.month == 1]
dataset = filter

print("\nVendas do mês de Janeiro:\n", dataset)
