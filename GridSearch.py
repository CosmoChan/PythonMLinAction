#!/usr/bin/env python
# encoding: utf-8

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

#Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Pipeline
pipe_svm = Pipeline([('scl',StandardScaler()),
                     ('clf', SVC(random_state = 1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = [{'clf__C':param_range,
               'clf__kernel':['linear']},
              {'clf__C':param_range,
               'clf__gamma':param_range,
               'clf__kernel':['rbf']}]

gs = GridSearchCV(estimator = pipe_svm,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)

gs.fit(X_train, y_train)

print gs.best_score_
print gs.best_params_

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print "Test accuracy: %.4f"%clf.score(X_test, y_test)

