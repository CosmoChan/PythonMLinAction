#!/usr/bin/env python
# encoding: utf-8

import sys
sys.path.append('~/文档/PythonMLinAction/pdr.py')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from pdr import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler





plt.figure(0)
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c = 'b', marker = 'x',label = '1')
plt.scatter(X_xor[y_xor == -1, 0],X_xor[y_xor == -1, 1], c = 'r', marker = 's', label = '-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

plt.figure(1)
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 1.0, C = 10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.title('Kernel SVM')
plt.legend(loc = 'upper left')
plt.show()

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

svmIri = SVC(kernel = 'rbf', random_state = 0, gamma = 0.2, C = 1.0)
svmIri.fit(X_train_std, y_train)

plt.figure(2)
plot_decision_regions(X_combined_std, y_combined, classifier = svmIri, test_idx = range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel('petal width [standardized]')
plt.title('Small gamma')
plt.show()


svmIriLarge = SVC(kernel = 'rbf', random_state = 0, gamma = 100, C = 1.0)
svmIriLarge.fit(X_train_std, y_train)

plt.figure(3)
plot_decision_regions(X_combined_std, y_combined, classifier = svmIriLarge, test_idx = range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel('petal width [standardized]')
plt.title('Large gamma')
plt.show()
