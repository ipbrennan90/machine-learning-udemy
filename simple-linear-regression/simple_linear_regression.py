# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = 0)
#Fitting regressor object to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set results
#create a vector of predictions of test set salary values
#vectors of predictions of dependent variables (salary for this example)
y_pred = regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs. Experience (training)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

