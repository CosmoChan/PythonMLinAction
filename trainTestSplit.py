#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



#Fetching data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None) df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total fenols', 'Flavanoids', 'Nonflavanoids fenols', 'Proanthocyanins', 'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#To see how many kinds of wine are there
print "Class labels: ", df_wine['Class label'].unique()

print df_wine.head()

#split the features and the label
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Min Max Scaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#Standardize
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

lr = LogisticRegression(penalty = 'l1', C = 0.1)
lr.fit(X_train_std, y_train)

print 'Training accuracy: ', lr.score(X_train_std, y_train)
print "Test accuracy: ", lr.score(X_test_std, y_test)

print 'Intercept:', lr.intercept_

print 'Coefficient: ', lr.coef_

fig = plt.figure(1)

ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty = 'l1', C = 10**c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column], label = df_wine.columns[column + 1], color = color)

plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.title(r'weights vs C = $\frac{1}{\lambda}$')
ax.legend(loc = 'upper center', bbox_to_anchor = (0.2, 0.5), ncol = 1, fancybox = True)
plt.show()



