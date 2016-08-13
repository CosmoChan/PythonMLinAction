#!/usr/bin/env python
# encoding: utf-8

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import  numpy as np
from pdr import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz


iris = load_iris()
X, y = iris.data[:,[2,3]], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)

tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier = tree, test_idx = range(105,150))
plt.xlabel('pental length [cm]')
plt.ylabel('pental width [cm]')
plt.title('Decision Tree Decision Regions')
plt.legend(loc = 'upper left')
plt.show()

export_graphviz(tree, out_file = 'tree.dot', feature_names = ['petal length', 'petal width'])

