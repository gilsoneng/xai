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