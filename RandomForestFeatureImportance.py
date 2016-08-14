#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#Fetching data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total fenols', 'Flavanoids', 'Nonflavanoids fenols', 'Proanthocyanins', 'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#Splitting data
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)

forest.fit(X_train, y_train)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]#descent sort

for f in range(X_train.shape[1]):
    print "%2d) %-*s %f"%(f+1, 30, feat_labels[f], importances[indices[f]])


plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], color = 'lightblue', align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

X_selected = forest.transform(X_train, threshold = 0.15)
print X_selected.shape




