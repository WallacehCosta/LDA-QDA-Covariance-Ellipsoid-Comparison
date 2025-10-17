import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Gerando um conjunto de dados fictício
np.random.seed(42)
X = np.random.rand(50, 5)  # 50 amostras com 5 variáveis

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para reduzir para 3 componentes principais
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Criando um gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotando os dados transformados
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='b', marker='o', alpha=0.7)

# Rótulos dos eixos
ax.set_xlabel('PC1 (1º Componente Principal)')
ax.set_ylabel('PC2 (2º Componente Principal)')
ax.set_zlabel('PC3 (3º Componente Principal)')

# Título
ax.set_title('Visualização 3D dos Dados Após PCA')

# Mostrar o gráfico
plt.show()
