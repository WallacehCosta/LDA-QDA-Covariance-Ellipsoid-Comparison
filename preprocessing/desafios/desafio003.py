import numpy as np
import pandas as pd

dataset = pd.read_csv("desafio003.csv")

x = dataset.drop(dataset["comprou"])
y = dataset["comprou"]

print(f"Variáveis independentes:\n", x)
print(f"\nVariáveis dependentes:\n", y)
