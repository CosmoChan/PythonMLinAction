#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder


df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                    ['red', 'L', 13.5, 'class2'],
                    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

print df
print "TRACK ================= "

size_mapping = {
    'XL':3,
    'L':2,
    'M':1
}

df['size'] = df['size'].map(size_mapping)
print df
print "TRACK ================= "

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print class_mapping
print "TRACK ================= "

df['classlabel'] = df['classlabel'].map(class_mapping)
print df
print "TRACK ================= "

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print y
print "TRACK ================="


