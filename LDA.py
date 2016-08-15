#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


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

#Calculating mean vectors
np.set_printoptions(precision = 4)

mean_vecs = []

for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis = 0))
    print "MV %s : %s \n"%(label, mean_vecs[label - 1])

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X[y == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print "Within class scatter matrix : %sx%s \n"%(S_W.shape[0], S_W.shape[1])

mean_overall = np.mean(X_train_std, axis = 0)
d = 13
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X[y == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print "Between-Class scatter matrix: %s x %s"%(S_B.shape[0], S_B.shape[1])

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True)
print 'Eigenvalues in decreasing order: \n'
for pair in eigen_pairs:
    print pair[0]

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse = True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha = 0.5, align = 'center', label = 'Individual discriminability')
plt.step(range(1, 14), cum_discr, where = 'mid', label = 'cummulative discriminability')
plt.xlabel('Linear Discriminants')
plt.ylabel('Discriminability ratio')
plt.ylim([-0.1, 1.1])
plt.legend(loc = 'best')
plt.show()

w= np.hstack((eigen_pairs[0][1][:,np.newaxis].real,
             eigen_pairs[1][1][:, np.newaxis].real))

print "Matrix: \n", w

X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c = c, label = l, marker = m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.title('LDA')
plt.legend(loc = 'best')
plt.show()

