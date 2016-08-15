#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

#Label encoding
print np.unique(y)
le = LabelEncoder()
y = le.fit_transform(y)
print np.unique(y)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Building pipeline
pipe_lr = Pipeline([('scl',StandardScaler()),
                    ('pca',PCA(n_components =2)),
                    ('clf', LogisticRegression(random_state = 0))
                    ])

#fit pipeline
pipe_lr.fit(X_train, y_train)

print "Test Accuracy: %0.4f"%pipe_lr.score(X_test, y_test)

#K Fold CV(Stratified)
k_fold = StratifiedKFold(y = y_train, n_folds = 10, random_state = 1)
scores = []

for k, (train, test) in enumerate(k_fold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print "Fold %s, Class dist: %s, Accuracy:%.3f"%(k+1, np.bincount(y_train[train]), score)
print "CV accuracy: %s +/- %s"%(np.mean(scores), np.std(scores))

#cross validation score
scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print "CV accuracy: %s"%scores
print "CV accuracy: %.3f +/- %.3f" %(np.mean(scores), np.std(scores))

