# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

# Importing the dataset
dataset = pds.read_csv("8 - Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# Training the linear regression on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualing the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualizing the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polymonial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualizing the polynomial regression results (higher resolution and smoother curve)

# Predict a new result with linear regression

# Predict a new result with polynomial regression