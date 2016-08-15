#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append('/home/cosmo/文档/PythonMLinAction.py')

from sklearn.linear_model import LogisticRegression
from sklearn import  datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import  train_test_split
import numpy as np
from pdr import plot_decision_regions
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix



iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

lr = LogisticRegression(C = 1000.0, random_state = 0)
lr.fit(X_train, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = lr, test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()


weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C = 10 ** c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label = 'petal length')
plt.plot(params, weights[:, 1], linestyle = '--', label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()

