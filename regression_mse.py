# Cross Validation Regression MSE
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'C:/Users/Sudheesh Nandakumar/Desktop/Data Science/Python_for Data Science/Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)

array = dataframe.values
X = array[:,0:13]
Y = array[:,13]

num_folds = 10
kfold = KFold(n_splits=10, random_state=7)

model = LinearRegression()

scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
results.mean()
results.std()

np.sqrt(-results.mean())
