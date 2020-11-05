# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:21:07 2020

@author: gilen
"""
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
df = pd.read_csv('winequality-red.csv') # Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# The target variable is 'quality'.
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
X_featurenames = X.columns

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                    feature_names=X_featurenames, 
                    class_names=['quality'], 
                    # categorical_features=, 
                    # There is no categorical features in this example, otherwise specify them.                               
                    verbose=True, mode='regression')

i = 278
exp = explainer.explain_instance(X_test.iloc[i], model.predict)
exp.as_pyplot_figure()

pd.DataFrame(exp.as_list())

X_test.iloc[i]

exp = explainer.explain_instance(X_test.iloc[i], model.predict)
exp.show_in_notebook(show_table=True, show_all=False)