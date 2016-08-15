#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from pdr import plot_decision_regions



#Fetching data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total fenols', 'Flavanoids', 'Nonflavanoids fenols', 'Proanthocyanins', 'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#split the features and the label
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Standardize
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

#Initializing LDA and fit the training data
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

#LR training
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)

#Plot the decision regions of training set
plot_decision_regions(X_train_lda, y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.title('LDA on training set')
plt.legend(loc = 'best')
plt.show()

#Plot the decision regions of test set
X_test_lda = lda.fit_transform(X_test_std, y_test)
plot_decision_regions(X_test_lda, y_test, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.title('LDA on test set')
plt.legend(loc = 'best')
plt.show()

