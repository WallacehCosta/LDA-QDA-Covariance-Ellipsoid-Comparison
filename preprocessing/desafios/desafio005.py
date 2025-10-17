import pandas as pd
import numpy as np

dataset = pd.read_csv("desafio005.csv")
print(dataset)

filter = dataset.loc[(dataset["preço"] > 100) & (dataset["estoque"] > 10)]
dataset2 = filter

print("\n\n", dataset2)
