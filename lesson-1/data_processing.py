#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:38:42 2017

@author: ianbrennan
"""

#Data Processing lesson

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split




#Importing dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
#using imputer library to replace missing values with feature means
#imputerObj = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
##fitting our columns with missing data with the imputer mean strategy
#imputerObj.fit(X[:, 1:3])
##transforming dataset
#X[:, 1:3] = imputerObj.transform(X[:, 1:3])
##encoding labels to help machine
#labelEncoder_X = LabelEncoder()
##encoding first column of x 
#X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
##need to encode again to make them dummy variables so they don't encode to categorical
#oneHotEncoder = OneHotEncoder(categorical_features = [0])
#X = oneHotEncoder.fit_transform(X).toarray()
#labelEncoder_y = LabelEncoder()
#Y = labelEncoder_y.fit_transform(Y)
# deciding your test train split, this gives or test %20 of our total sample size

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Feature scaling, making sure the euclidean distance accurately reflects scale, this can make one feature have no significance
"""
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
#do not need to feature scale your dependent variable when it is categorical, it only has a value of 0 or 1
"""

