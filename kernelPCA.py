#!/usr/bin/env python
# encoding: utf-8


from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation

    X: shape = [n_samples, n_features]
    gamma: float, tuning parameters of RBF kernel

    n_components: int, number of principal components to return
    """

    #Calculating pairwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')

    #Convert pairwise distances into square matrix
    mat_sq_dists = squareform(sq_dists)

    #Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Obtaining eigen pairs of K
    eigvals, eigvecs = eigh(K)

    #Collect the top k eigenvectors(objected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt



X, y = make_moons(n_samples = 100, random_state = 123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], marker = '^', color = 'red', alpha = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker = 'o', color = 'blue', alpha = 0.5)
plt.show()

#What will happen if we use the naive PCA?
from sklearn.decomposition import  PCA

scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
             color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.show()

#And how about our RBF kernel PCA?
from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
              color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
              color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
             color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

#Using aother data set
from sklearn.datasets import make_circles
X, y = make_circles(n_samples = 1000, random_state = 123, noise = 0.1, factor = 0.2)


plt.scatter(X[y == 0, 0], X[y == 0, 1], marker = '^', color = 'red', alpha = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker = 'o', color = 'blue', alpha = 0.5)
plt.show()

#scikit learn PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
             color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.show()


#RBF kernel PCA
X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
              color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
              color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
             color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()
