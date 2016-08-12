#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import  load_iris
from matplotlib.colors import  ListedColormap

class AdalineGD(object):
    """Adaptive Linear Neuron Classifier

    Parameters
    ------------
    eta: float
        Learning rate between 0 and 1
    n_iter: int
        Passes over the training dataset

    Attributes
    ------------
    w_: 1d-array
        weights after fitting
    errors_:list
        Number of misclassification in every epoch
    """

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        -------------
        X:{array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """

        self.w_ = np.zeros(1 + X.shape[1]) #Add w_0
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)#analoge ? : in C++

def plot_decision_region(X, y, classifier, resolution = 0.02):
        #setop marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        #plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        #plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                       alpha = 0.8, c = cmap(idx),
                       marker = markers[idx], label = cl)
"""
Loading iris data
"""
df = load_iris()
X, y = df.data, df.target
X = X[:100, [0, 2]]
y = y[:100]

"""
It is not a good idea to set the learning rate too large or too small
"""
plt.figure(1)
ax1 = plt.subplot(121)
ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax1.plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax1.set_xlabel('Epoches')
ax1.set_ylabel('log(Sum-squaered-error)')
ax1.set_title('Adaline-Learning rate 0.01')

ax2 = plt.subplot(122)
ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
ax2.plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax2.set_xlabel('Epoches')
ax2.set_ylabel('Sum-squared-error')
ax2.set_title('Adaline-Learning rate 0.0001')
plt.show()

"""
Standardized data will speed up the learning process
"""
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)

plot_decision_region(X_std, y, classifier = ada)

plt.title('Adaline-Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epoches')
plt.ylabel('Sum-squared-error')
plt.show()
