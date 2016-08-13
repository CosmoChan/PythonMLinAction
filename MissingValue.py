#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = """A,B,C,d
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,"""

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

print "head:"
print df.head()

print "isnull.sum:"
print df.isnull().sum()

print "dropna:"
print df.dropna()

print "dropna(axis = 1)"
print df.dropna(axis = 1)

print "dropna(how = all)"
print df.dropna(how = 'all')

print "Drop the rows which have less than 4 nonzero value"
print df.dropna(thresh = 4)

print "Drop subset"
print df.dropna(subset = ['C'])

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr.fit(df)
imputed_data = imr.transform(df.values)

print df.values
print "TRACK ================= "
print imputed_data


