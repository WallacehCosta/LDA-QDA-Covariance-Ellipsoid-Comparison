import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

array = np.array([
    [1, 2],
    [2, 4],
    [4, 8],
    [8, 16],
    [16, 32]
])
print(array)

#média dos features
mean = np.mean(array, axis=0)
print(f"\nMédia dos features: {mean}")

#calculando o desvio padrão
std = np.std(array, axis=0, ddof=1) #Não questiona o uso do 'ddof=1'
print(f"Desvio padrão: {std}")
z = (array - mean) / std

print("\n", z)

print(f"\n", np.mean(z, axis=0))
print(np.std(z, axis=0, ddof=1))
