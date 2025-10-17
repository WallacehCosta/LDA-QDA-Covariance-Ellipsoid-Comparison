import pandas as pd

dataset = pd.read_csv("desafio006.csv")
print("Antes:\n", dataset)

dataset = dataset.rename(columns= {"Age": "Idade", "Name": "Nome", "Salary": "Salário"})

print("\nDepois:\n", dataset)
