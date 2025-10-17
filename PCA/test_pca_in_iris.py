from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_iris()
x = data.data
y = data.target

x_scaler = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaler)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA - Iris Dataset")
plt.show()
