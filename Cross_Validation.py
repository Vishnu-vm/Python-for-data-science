# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:48:33 2020

@author: Vishnu Mohan
"""


# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'C:/Users/Sudheesh Nandakumar/Desktop/Data Science/Python_for Data Science/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression(max_iter=500)

results = cross_val_score(model, X, Y, cv=kfold)


#Print results

results.mean()*100.0
results.std()*100.0
