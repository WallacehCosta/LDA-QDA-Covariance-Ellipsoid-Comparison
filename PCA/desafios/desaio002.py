import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

array = np.array([
    [2, 3.2],
    [3.2, 1.7],
    [0.75, 6.3],
    [6, 7.3],
    [10, 7]
])

scaler = StandardScaler()
std_scaler = scaler.fit_transform(array)

pca = PCA()
x_pca = pca.fit_transform(std_scaler)

print(pca.explained_variance_ratio_)

cum = np.cumsum(std_scaler, axis=0)
print(cum)
