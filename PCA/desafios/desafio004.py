from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#import numpy as np

data = load_iris()
x = data.data
y = data.target

x_sclt = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_sclt)
print(pca.explained_variance_ratio_)

x_reconstructed = pca.inverse_transform(x_pca)

mean = mean_squared_error(x_sclt, x_reconstructed) #Comparar com dados escalados
print("Erro médio da reconstrução: ", mean)

#Visualizar a diferença
'''
x_reconstructed_original = StandardScaler().fit(x).inverse_transform(x_reconstructed)
plt.figure(figsize=(8, 4))
plt.plot(np.arange(x.shape[0]), x[:, 0], label='Original (feature 0)')
plt.plot(np.arange(x.shape[0]), x_reconstructed_original[:, 0], label='Reconstruído (feature 0)', linestyle='--')
plt.legend()
plt.title("Comparação original vs reconstruído (feature 0)")
plt.xlabel("Amostras")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
'''
