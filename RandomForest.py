#!/usr/bin/env python
# encoding: utf-8

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pdr import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
import time



iris = load_iris()
X, y = iris.data[:,[2,3]], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_combined, y_combined = np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

start = time.clock()

forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs = 2)

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier = forest, test_idx = range(105, 150))
plt.xlabel("petal length [cm]")
plt.ylabel('petal width [cm]')
plt.title('Random Forest')
plt.show()

print "Time used: %0.6f"%(time.clock() - start)
