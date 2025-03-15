#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

# importing the dataset
dataset = pds.read_csv("6 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values
print("Years of experience: ", X)
print("Salary: ", y)

# Splitting dataset into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Dependent training variable: ", X_train)
print("Dependent test variable: ", X_test)
print("Independent training variable: ", y_train)
print("Dependent test variable: ", y_train)

# Training the simple regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test results
y_pred = regressor.predict(X_test)

# visualizing training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# visualizing test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()