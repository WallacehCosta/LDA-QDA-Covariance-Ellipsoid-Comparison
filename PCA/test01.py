import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset = np.array([[0.21, 0.5], [0.76, 0.13], [0.1, 0.98]])

scaler = StandardScaler()
dst_scaler = scaler.fit_transform(dataset)

pca = PCA() #Para ver quanta variância possui, esse parâmetro precisa estar vazio
x_pca = pca.fit_transform(dst_scaler)

print(x_pca)
print(pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Número de Componentes")
plt.ylabel("% variância explicada")
plt.grid(True)
plt.show()

'''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Gerando dados fictícios (exemplo simples)
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])
print(X)

# Centralizar os dados (normalização)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n\nArray normalizado:\n", X_scaled)

# Aplicar PCA
pca = PCA(n_components=1)  # Reduzir para 1 componente principal
X_pca = pca.fit_transform(X_scaled)
print("\n\nRedução pós PCA:\n", X_pca)

print("\n\nVariância explicada por cada componente:\n", pca.explained_variance_ratio_)

# Visualizando os dados transformados
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], color='blue', label='Dados originais')
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), color='red', label='PCA projetado')
plt.legend()
plt.show()
'''
