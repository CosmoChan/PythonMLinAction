#!/usr/bin/env python
# encoding: utf-8

import sys
sys.path.append('home/文档/PythonMLinAction/pdr.py')
from pdr import plot_decision_regions
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import  datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import  train_test_split
import numpy as np


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


svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

from sklearn.linear_model import SGDClassifier
SGDsvm = SGDClassifier(loss = 'hinge')
SGDsvm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = SGDsvm, test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('pental width [standardized]')
plt.legend(loc = 'upper left')
plt.title('SGD SVM')
plt.show()

