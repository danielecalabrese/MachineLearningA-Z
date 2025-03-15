#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

# importing the dataset
dataset = pds.read_csv("7 - Multiple Linear Regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encorder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting dataset into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate())



