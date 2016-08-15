#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

#Label encoding
print np.unique(y)
le = LabelEncoder()
y = le.fit_transform(y)
print np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty = 'l2', random_state = 0))
                    ])

train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr,
                                                       X = X_train,
                                                       y = y_train,
                                                       train_sizes = np.linspace(0.1, 1.0, 10),
                                                       cv = 10,
                                                       n_jobs = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(train_sizes, train_mean,color = 'blue', marker = 'o', markersize = 5, alpha = 0.5, label = 'Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean,color = 'green', linestyle = '--', marker = '^', markersize = 5, alpha = 0.5, label = 'Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean- test_std, alpha = 0.15, color = 'green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.title('Learning Curve')
plt.legend(loc = 'lower right')
plt.show()


