#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#import dataset with pandas
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#splitting dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)
