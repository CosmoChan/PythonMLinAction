#!/usr/bin/env python
# encoding: utf-8

from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt


X, y = make_moons(n_samples = 1000, random_state = 123)

scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_skern_pca = scikit_kpca.fit_transform(X)

plt.scatter(X_skern_pca[y == 0, 0], X_skern_pca[y == 0, 1],
            color = 'red', marker = '^', alpha = 0.2)
plt.scatter(X_skern_pca[y == 1, 0], X_skern_pca[y == 1, 1],
            color = 'blue', marker = 'o', alpha = 0.2)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


