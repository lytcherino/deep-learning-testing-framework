a# -*- coding: utf-8 -*-

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

############################

data = DataHandler()
data.loadData('data_updated.csv', colRemove = 7, skiprows = 0)

X, _, y, _= data.getSplitData(normalizeData = False, testSplit = 0)
features = data.features
feature_columns = data.feature_columns

#############
## PCA
#############

from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X)

lower_dimensional_data = pca.transform(X)

total_variance = sum(pca.explained_variance_)

var_explained = [(i/total_variance)*100 for i in sorted(pca.explained_variance_, reverse=True)]

cum_var = np.cumsum(var_explained)

plt.figure()
size = np.size(pca.explained_variance_)
plt.scatter(np.arange(1,size+1,1), cum_var, color ='b')
#plt.step(np.arange(0,size,1), cum_var, where= 'mid', color ='b')
plt.axhline(y = 95, color = 'c', linestyle = '--', label='95%')
plt.axhline(y = 90, color = 'g', linestyle = '--', label='90%')
plt.axhline(y = 85, color = 'k', linestyle = '--', label='85%')
plt.xticks(np.arange(1,features+1,1))

plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.legend()

plt.show()

#####################



#####################
## Linear Regression
#####################

linear = R.LinearRegression()
X_train, X_test, y_train, y_test = data.getSplitData()
linear.train(features, X_train, X_test, y_train, y_test, n_jobs = 1, verbose = True, startIndex = 1)
linear.fit(X, y)
#func = linear.function(columnNames=['D','E', 'F', 'G', 'L', 'P', 'U', 'AA', 'AB', 'AD'], featureStartIndex = 3)
#func = linear.function(columnNames=['D','E', 'F', 'G', 'P','W','X','Y','AA', 'AB', 'AD'], featureStartIndex = 3)

linear.function(columnNames=[feature_columns[letter-ord('A')] for letter in range(ord('A'), ord('A') + features)], featureStartIndex = 2);
#linear.function(columnNames=[chr(letter) for letter in range(ord('A'), ord('A') + features)], featureStartIndex = 2);

#func = linear.function(columnNames=['A','B','C','D','E','F','G'], featureStartIndex = 2)
#func = linear.function(columnNames=['D', 'E', 'F', 'G', 'H','I','J','K', 'L','M','N','O', 'P', 'U','W','X','Y', 'AA', 'AB', 'AD'], featureStartIndex = 3)

y_pred_linear, -1.0*cross_score_poly = data.predict(linear)

linear_analyser = DataAnalyser(R.LinearRegression(), feature_columns)

linear_analyser.run(data, startPercentage = 10, repeats = 3, verbose = True)
linear_analyser.plotMse(axis=(25,140,0,0.2))
linear_analyser.plotMseStd(axis=(25,140,0,0.2))
linear_analyser.plotParameterCount()
linear_analyser.plotParameterProbabilities()
#l = linear_model.LinearRegression()
#l.fit(X_train, y_train)
#l.coef_
#l.intercept_
#y_pred = l.predict(X_test)

#########################
# Polynomial Regression
#########################


polyReg = R.PolynomialRegression()
X_train, X_test, y_train, y_test = data.getSplitData(normalizeData = False, validationData = False)
polyReg.train(features, X_train, X_test, y_train, y_test, exit_early=True, verbose = True, cv = 10, startIndex = 1)
polyReg.fit(X, y, cv = 10)

#polyReg.function(columnNames=[chr(letter) for letter in range(ord('A'), ord('A') + features)], featureStartIndex = 2);
polyReg.function(columnNames=[feature_columns[letter-ord('A')] for letter in range(ord('A'), ord('A') + features)], featureStartIndex = 2);

print(polyReg.results)

data_1 = DataHandler()
data_1.loadData('predict_example_new_long.csv', colRemove = 7, skiprows = 1)
X_, _, y_, _= data_1.getSplitData(normalizeData = False, testSplit = 0)
polyReg.fit(X_, y_, cv = 10)


y_pred_poly, cross_score_poly = data.predict(polyReg, X = X_, y = y_)
print('MSE: ', np.sum((y_-y_pred_poly**2)/len(y_)))

analyser = DataAnalyser(R.PolynomialRegression(), feature_columns)
analyser.run(data, startPercentage = 25)

analyser.plotMse(axis=(25,140,0,0.2))
analyser.plotMseStd(axis=(25,140,0,0.2))
analyser.plotParameterCount()
analyser.plotParameterProbabilities()


#plt.scatter(np.arange(0,np.size(y_pred),1), y_pred, label='Predicted')
#plt.scatter(np.arange(0,np.size(y_test),1), y_test, label='Actual')

plt.scatter(np.arange(1,10,0.01), y_scaler.inverse_transform(y_pred), label='Predicted', color = 'r')
plt.scatter(np.arange(1,10,0.01), y_, label='Actual', color = 'g')


###########
# ANN
###########

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataHandler import DataHandler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, RepeatedKFold

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import callbacks

import datetime
import re

import os
import errno

import seaborn as sns

import NNTestingFramework as NNTF


max_hidden_units = 16
max_dropout = 1.0
X_train, X_test, y_train, y_test = data.getSplitData(validationData = False, normalizeData = True, testSplit = 0.2)
NN = NNTF.TestingFramework(NNTF.ModelBuilder.hidden_dropout)
NN.train(np.arange(1, max_hidden_units + 1),np.arange(0.0, max_dropout, 0.1), X_train, y_train, X_test, y_test)
NN.plot(scale = 3.0)
X, _, y, _= data.getSplitData(normalizeData = True, testSplit = 0)
NN.evaluate(X, y, cv = 10)

max_hidden_units = 2
max_hidden_layers = 2
max_dropout = 0.1

data = DataHandler()
data.loadData('data_updated.csv', colRemove = 7)
X_train, X_test, y_train, y_test, X_val, y_val = data.getSplitData(validationData = True, normalizeData = True)
NN_0 = NNTF.TestingFramework(NNTF.ModelBuilder.hidden_layers)

NN_0.train(X_train, y_train, X_test, y_test, 
           patience = 10,
           param_1 = {'Hidden Units': np.arange(1, max_hidden_units + 1)}, 
           param_2 = {'Dropout': np.arange(0, max_dropout, 0.1)},
           param_3 = {'Hidden Layers': np.arange(0, max_hidden_layers, 1)})

NN_0.plotAll(scale = 4.0)
#X, _, y, _= data.getSplitData(normalizeData = True, testSplit = 0)
NN_0.evaluate(X_val, y_val, cv = 10, verbose = True)


# 7, 11, 15, 18, 25
data.loadData('data_updated.csv', colRemove = 18)


data = DataHandler()
data.loadData('mos_data_extended.csv', colRemove = -1, skiprows = 1)
res = data.pearson()

max_hidden_units = 30
max_hidden_layers = 6
max_dropout = 0.6

NN_0 = NNTF.TestingFramework(NNTF.ModelBuilder.hidden_layers_drop_same)
NN_0.gridsearch(data, 
                param_0 = {'Hidden Units': np.arange(1, max_hidden_units + 1)}, 
                param_1 = {'Dropout': np.arange(0, max_dropout, 0.1)},
                param_2 = {'Hidden Layers': np.arange(3, max_hidden_layers, 1)})



# Testing FFNN model with increasing amounts of training examples
    
start = 10
end = 100
step = 1

elements = ((end - start + 1) // step) + 1

prediction_errors_test = np.zeros(elements)
prediction_errors_test_error = np.zeros(elements)

prediction_errors_train = np.zeros(elements)
prediction_errors_train_error = np.zeros(elements)

prediction_errors_val = np.zeros(elements)
prediction_errors_val_error = np.zeros(elements)
prediction_errors_val_error_std = np.zeros(elements)

num_epochs = np.zeros(elements)
num_epochs_error = np.zeros(elements)

data = DataHandler()
data.loadData('data_updated.csv', colRemove = 7)

loops = 10
hidden_units = 17
hidden_layers = 1
dropout = 0.0
patience = 100

title = 'Hidden Units: {}, Hidden Layers: {}, Dropout: {}'.format(hidden_units, hidden_layers, dropout)

for i in range(start, end + 1, step):
    print(str(i) + ': ', end = '')

    train_results = np.array([])
    test_results = np.array([])
    val_results = np.array([])
    val_results_error = np.array([])
    num_epoch_results = np.array([])
    
    index = (i-start)//step

    for j in range(loops):
        print(str(j) + ' ', end='')
        NN_1 = NNTF.TestingFramework(NNTF.ModelBuilder.hidden_layers_drop_same, baseDirectory = './ANN/Performance')
        
        X_train, X_test, y_train, y_test, X_val, y_val = data.getTruncatedDataLessTraining(i, validationData = True, normalizeData = True)
        
        fitHistory = NN_1.train(X_train, y_train, X_test, y_test, 
                               patience = patience,
                               returnFitHistory = True,
                               param_0 = {'Hidden Units': np.arange(hidden_units, hidden_units + 1)}, 
                               param_1 = {'Dropout': np.arange(dropout, dropout + 0.1, 0.1)},
                               param_2 = {'Hidden Layers': np.arange(hidden_layers, hidden_layers + 1, 1)}   
                               )
        
        best_epoch = np.array(fitHistory['val_loss']).argmin()
        
        train_results = np.append(train_results, fitHistory['loss'][best_epoch])
        test_results = np.append(test_results, fitHistory['val_loss'][best_epoch])
        
        evalData = NN_1.evaluateAll(X_val, y_val, cv = 10, verbose = False)
        num_epoch_results = np.append(num_epoch_results, len(fitHistory['loss']))
        
        for score in evalData:
            val_results = np.append(val_results, score[1][0])
            val_results_error = np.append(val_results_error, score[1][1])
        
    print()
    
    prediction_errors_test[index] = test_results.mean()
    prediction_errors_test_error[index] = test_results.std()

    prediction_errors_train[index] = train_results.mean() 
    prediction_errors_train_error[index] = train_results.std() 

    prediction_errors_val[index] = val_results.mean()
    prediction_errors_val_error[index] = val_results_error.mean()
    prediction_errors_val_error_std[index] = val_results.std()

    num_epochs[index] = num_epoch_results.mean()
    num_epochs_error[index] =num_epoch_results.std()

plt.figure()
plt.scatter(np.arange(start, end + 1, step), prediction_errors_test, color = 'blue')
plt.errorbar(np.arange(start, end + 1, step), prediction_errors_test, yerr = prediction_errors_test_error, label = 'Test', color = 'blue')

plt.errorbar(np.arange(start, end + 1, step), prediction_errors_train, yerr = prediction_errors_train_error, label = 'Train', color = 'green')
plt.scatter(np.arange(start, end + 1, step), prediction_errors_train, color = 'green')

plt.errorbar(np.arange(start, end + 1, step), prediction_errors_val, yerr = prediction_errors_val_error_std, label = 'Val', color = 'red')
plt.scatter(np.arange(start, end + 1, step), prediction_errors_val, color = 'red')

plt.xlabel('Training Data Used (%)')
plt.ylabel('MSE')
plt.axis([0, 100, 0, 0.3])
plt.title(title)
plt.legend()

#plt.figure()
#plt.scatter(np.arange(start, end + 1, step), prediction_errors_test-prediction_errors_train, color = 'green')
#plt.plot(np.arange(start, end +1, step), prediction_errors_test-prediction_errors_train, color = 'green')

plt.figure()
plt.errorbar(np.arange(start, end + 1, step), num_epochs, yerr = num_epochs_error, label = 'Epochs', color = 'blue')
plt.scatter(np.arange(start, end + 1, step), num_epochs, color = 'blue')
plt.xlabel('Training Data Used (%)')
plt.ylabel('Epochs Until Early Stopped')
plt.title(title)

#NNTF.TestingFramework.evaluateModelData(model, testData, trainingData, colRemove=7)
#NNTF.TestingFramework.predict(model, testData, trainingData, colRemove=7)

NNTF.TestingFramework.evaluateAllModelsData('./ANN/Output/2018-07-06-17-46-15/', testData, trainingData,
                                            params = ['Hidden Units', 'Dropout',' Hidden Layers'],
                                            colRemove = 7)
    
    

seed = 1
np.random.seed(seed)


start = 25
end = 100
prediction_errors_test = np.zeros(end-start)
prediction_errors_train = np.zeros(end-start)

minimum_error = 1
minimum_epoch = 0

#X_train, X_test, y_train, y_test, X_val, y_val = data.getSplitData(validationData = True, normalizeData = True)

for i in range(start, end):
    X_train, X_test, y_train, y_test = data.getTruncatedDataLessTraining(validationData = False, normalizeData = True)
    model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    
    y_pred_inv = y_scaler.inverse_transform(y_pred).reshape(-1,1)
    y_val_inv = y_scaler.inverse_transform(y_val).reshape(-1,1)
    
    y_pred_train_inv = y_scaler.inverse_transform(y_pred_train).reshape(-1,1)
    y_train_inv = y_scaler.inverse_transform(y_train).reshape(-1,1)

    
    pred_error_val = metrics.mean_squared_error(y_pred_inv, y_val_inv)
    pred_error_train = metrics.mean_squared_error(y_pred_train_inv, y_train_inv)
    
    model = KerasRegressor(build_fn = createModel, verbose = 0, batch_size = 16, epochs = i)
    results = cross_val_score(model, X_train, y_train, cv = 3, n_jobs = 1, verbose = 0)
    
    
    if pred_error_test < minimum_error:
        minimum_epoch = i
        minimum_error = pred_error_test

    prediction_errors_test[i-start] = pred_error_test
    prediction_errors_train[i-start] = pred_error_train

    print("Epochs : ", i, "(Test) Prediction Error: ", pred_error_test)
    print("Epochs : ", i, "(Train) Prediction Error: ", pred_error_train)
    
print('Minimum test prediction MSE of, ', minimum_error, ' for epoch: ', minimum_epoch)
    
plt.figure()
plt.xlabel('Epochs Trained')
plt.ylabel('MSE')
prediction_errors_test = prediction_errors_test[prediction_errors_test != 0]
prediction_errors_train = prediction_errors_train[prediction_errors_train != 0]

plt.scatter(np.arange(start,len(prediction_errors_train)+start,1), prediction_errors_train, label ='Train')
plt.scatter(np.arange(start,len(prediction_errors_test)+start,1), prediction_errors_test, label ='Test')
plt.legend()


#plt.figure()
#plt.scatter(y_test_inv, y_pred_inv, color = 'b', label ='predicted')
#plt.show()

pred_error = metrics.mean_squared_error(y_pred_inv, y_test_inv)


model = KerasRegressor(build_fn = createModel, verbose = 0, batch_size = 16, epochs = 25)

results = cross_val_score(model, X_train, y_train,
                          cv = 10, n_jobs = 1, verbose = 0)

# Grid Search
optimizers = ['adam']
activations = ['relu']
init = ['glorot_uniform']
batches = [4,8,16,32]
epochs = [50,100, 200, 250,500]
dropout_rates = [0.0, 0.1, 0.2, 0.3]

params = dict(optimizer = optimizers, epochs = epochs, batch_size = batches, init = init, dropout_rate=dropout_rates, activation = activations)

cv = RepeatedKFold(n_splits = 5, n_repeats = 10)
grid = GridSearchCV(estimator=model, param_grid = params, cv = cv)

grid_result = grid.fit(X_train, y_train)

print("Best Result: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                                    
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) for %r" % (mean, stdev, param))
    
    
##
###########


##################################
## Multivariate Regression Splines
##################################

from pyearth import Earth
from sklearn import metrics
from matplotlib import pyplot as plt

model = Earth(verbose = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_pred, y_test)
evs = metrics.explained_variance_score(y_pred, y_test)

y_pred_inv = y_scaler.inverse_transform(y_pred)
y_test_inv = y_scaler.inverse_transform(y_test)

plt.figure()
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Predicated')
plt.ylabel('Actual')
plt.axhline(y = 5, color = 'g', linestyle = '--', label='MOS = 5')
plt.axhline(y = 1, color = 'g', linestyle = '--', label='MOS = 1')


y = lambda x: x*1
x = np.linspace(np.min(y_pred_inv), np.max(y_pred_inv), len(y_pred_inv)/3)
y = y(x)

plt.plot(x,y, color = 'r', label='y=x')
plt.legend()

print("MSE: %.2f\nExplained Variance: %.2f" % (mse, evs))

 #Print the model
#print(model.trace())
#print(model.summary())
