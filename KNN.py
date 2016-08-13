#!/usr/bin/env python
# encoding: utf-8

from sklearn.neighbors import KNeighborsClassifier
from pdr import plot_decision_regions
from sklearn.datasets import  load_iris
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#Loading iris data
iris = load_iris()
X, y = iris.data[:,[2,3]], iris.target

#Splitting data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Combining data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

#Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#KNN model
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)

#Plotting Decision regions
plot_decision_regions(X_combined, y_combined, classifier = knn, test_idx = range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.title('KNN')
plt.show()





