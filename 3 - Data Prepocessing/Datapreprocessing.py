# Importing the libraries -> pip install matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv('3 - Data Prepocessing/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Indipendet variables: ", X)
print("Dipendet variables: ", y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("Indipendet variables trasformed: ", X)

# Encoding categorical data
# Encoding the Independent Variable -> trasforma le stringhe in array
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encorder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("Indipendet variables trasformed and encoded: ", X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("Dipendet variables trasformed and encoded: ", y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
print("Indipendet variables training set: ", X_train)
print("Indipendet variables test set: ", X_train)
print("Dipendet variables training set: ", y_train)
print("Dipendet variables test set: ", y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:]) # https://stackoverflow.com/questions/43675665/when-scale-the-data-why-the-train-dataset-use-fit-and-transform-but-the-te#:~:text=We%20use%20fit_transform()%20on,to%20scale%20the%20test%20data.











