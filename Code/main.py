#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
# Import all the libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import Regression as R
from DataHandler import DataHandler
from DataAnalyser import DataAnalyser

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import callbacks

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, RepeatedKFold

import NNTestingFramework as NNTF

import json

############################


# Feature Selection

# Univariate Selection

#from sklearn.feature_selection import RFE
#from sklearn import linear_model
#
#data = DataHandler()
#data.loadData('data_updated.csv', colRemove = -1)
#
#X, _, y, _= data.getSplitData(normalizeData = True, testSplit = 0)
#
#
#linear = linear_model.LinearRegression()
#rfe = RFE(linear, 4, verbose=1)
#fit = rfe.fit(X, y.flatten())
#features = [data.feature_columns[i] for i in list(np.where(fit.support_)[0])]
#print("Selected Features: {}".format(fit.support_))
#print("Feature Ranking: {}".format(fit.ranking_))
#
#print("Features: {}".format(", ".join(features)))



NNTF.TestingFramework.predict(model1, testData, trainingData, colRemove=7, skiprowsTest = 1, skiprowsTrain = 1, verbose = True)
NNTF.TestingFramework.predict(model_final, trainingDataExt, trainingDataExt, colRemove=8, skiprowsTest = 1, skiprowsTrain = 1, verbose = True)

NNTF.TestingFramework.predict(model2, trainingData, trainingData, colRemove=-1, skiprows = 0, verbose = False)

NNTF.TestingFramework.evaluateAllModelsData('./ANN/Output/', trainingData, trainingData,
                                            params = ['Hidden Units', 'Dropout',' Hidden Layers'],
                                            colRemove = 8, verbose = True, 
                                            skiprowsTrain = 1, skiprowsTest = 1)


data = DataHandler()
data.loadData('data_updated_parameters_long.csv', colRemove = -1, skiprows = 1)
data.saveNormParamsAsJson('normalization-parameters-extended.json')

if __name__ == '__main__':
    
    data = DataHandler()
    data.loadData('data_updated.csv', colRemove = 7)
    
    data.getSplitData(validationData = True, normalizeData = True)
    
    data.getNormParams(verbose = True)
    print(data.feature_columns)
