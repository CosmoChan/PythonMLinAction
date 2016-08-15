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



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

#Label encoding
print np.unique(y)
le = LabelEncoder()
y = le.fit_transform(y)
print np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

pipe_lr = Pipeline([('scl',StandardScaler()),
                    ('pca',PCA(n_components =2)),
                    ('clf', LogisticRegression(random_state = 0))
                    ])

pipe_lr.fit(X_train, y_train)

print "Test Accuracy: %0.4f"%pipe_lr.score(X_test, y_test)
