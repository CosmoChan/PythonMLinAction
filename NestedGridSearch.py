#!/usr/bin/env python
# encoding: utf-8


from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
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
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state = 1))])

#SVC nested cross validation
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = [{'clf__C':param_range,
               'clf__kernel':['linear']},
              {'clf__C':param_range,
               'clf__gamma':param_range,
               'clf__kernel':['rbf']}]
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)

#Note that the estimator in the follow line is gs, this is what 'nested' means
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)

print "CV accuracy: %.4f +/- %.4f"%(np.mean(scores), np.std(scores))

#Decision Tree Nested Cross Validation
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0),
                 param_grid = [{'max_depth':[1, 2, 3, 4, 5, 6, 7, None]}],
                 scoring = 'accuracy',
                 cv = 5)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)
print 'CV accuracy: %.4f +/- %.4f'%(np.mean(scores), np.std(scores))
