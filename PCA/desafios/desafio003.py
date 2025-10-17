from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_digits()
x = data.data
y = data.target

x_scaled = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print(pca.explained_variance_ratio_)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='tab10', s=15) #Ajuda a melhorar a visibilidade (cmap='tab10', s=15)
plt.colorbar() #Mete uma colobar do lado
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()
