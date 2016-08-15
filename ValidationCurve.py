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
from sklearn.learning_curve import validation_curve


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

#Label encoding
print np.unique(y)
le = LabelEncoder()
y = le.fit_transform(y)
print np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty = 'l2', random_state = 0))
                    ])


"""
Validation Curve
"""

param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_scores, test_scores = validation_curve(estimator = pipe_lr,
                                             X = X_train,
                                             y = y_train,
                                             param_name = 'clf__C',
                                             param_range = param_range,
                                             cv = 10)

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)

test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_range, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std,
                color = 'blue', alpha = 0.15)
plt.plot(param_range, test_mean,
         color = 'green', marker = '^',
         markersize = 5, label = 'Test accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std,
                 color = 'green', alpha = 0.15)
plt.xlabel('Parameter C')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend('lower right')
plt.ylim([0.8, 1.0])
plt.show()


